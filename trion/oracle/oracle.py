"""
Optimization-Sensitive Oracles (Section 2.5).

S(m) = S_diff(m) + Δ_opt(m)

S_diff:   cross-backend inconsistency   (pytorch_eager ref vs target)
Δ_opt:    optimization-induced deviation (target+opt vs target−opt)

Crashes are logged separately and do NOT contribute to S(m).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import onnx

from ..config import TrionConfig
from .base import BackendBase, BackendResult
from .pytorch_backend import PyTorchEagerBackend
from .onnxruntime_backend import ONNXRuntimeBackend
from .tvm_backend import TVMBackend
from .tensorrt_backend import TensorRTBackend
from .xla_backend import XLABackend
from .torchscript_backend import TorchScriptBackend
from .torch_compile_backend import TorchCompileBackend
from .tf_backend import TFBackend
from .tflite_backend import TFLiteBackend
from .openvino_backend import OpenVINOBackend

logger = logging.getLogger(__name__)

# Errors that are clearly NOT compiler bugs — they're ONNX→PyTorch conversion
# issues or torch.fx tracing limitations in onnx2torch, not the compiler itself.
_FRONTEND_KEYWORDS = [
    "onnx2torch",           # conversion library failure
    "FakeTensor",           # torch.compile tracing via onnx2torch calls torch.Size(FakeTensor)
    "fake_tensor",
    "make_fx",              # torch.fx graph capture failure
    "symbolic_trace",       # torch.fx symbolic tracing
    "torch.fx",
    "dynamo",               # TorchDynamo frontend
    "BackendCompilerFailed", # dynamo caught a backend error and wrapped it
]

# Errors that are genuine compiler/runtime bugs
_BACKEND_KEYWORDS = [
    "CUDA",                 # GPU kernel error
    "cudnn",                # cuDNN failure
    "RuntimeError",         # real execution failure (not tracing)
    "illegal memory",       # memory corruption
    "INTERNAL ASSERT",      # compiler assertion
    "lax.",                 # JAX/XLA primitive failure
    "conv_general",         # XLA convolution bug
    "shape mismatch",       # compiler shape inference bug
    "DType",                # dtype handling bug
    "Not implemented",      # missing op in compiler
]


def _classify_crash(error: str) -> str:
    """Classify crash as 'frontend' (not a compiler bug) or 'backend' (compiler bug).

    Decision logic:
    - If any frontend keyword is present → 'frontend' (conversion/tracing artifact)
    - Otherwise → 'backend' (genuine compiler failure)
    """
    if any(k in error for k in _FRONTEND_KEYWORDS):
        return "frontend"
    return "backend"


def _error_signature(error: str) -> str:
    """Extract a short normalized signature for grouping same-type errors."""
    import re
    # Take first non-empty line of the error
    first_line = next((l.strip() for l in error.splitlines() if l.strip()), error)
    # Normalize: remove memory addresses, line numbers, tensor shapes
    sig = re.sub(r'0x[0-9a-fA-F]+', '<addr>', first_line)
    sig = re.sub(r'\b\d+\b', 'N', sig)
    return sig[:120]


@dataclass
class OracleReport:
    """Per-model oracle result."""
    model_id: int
    total_score: float
    s_diff: Dict[str, float] = field(default_factory=dict)     # backend → score
    delta_opt: Dict[str, float] = field(default_factory=dict)  # backend → score
    crashes: List[str] = field(default_factory=list)           # crashed backend labels
    errors:  Dict[str, str] = field(default_factory=dict)      # backend → error msg
    crash_info: Dict[str, str] = field(default_factory=dict)   # backend → "frontend"|"backend"
    error_signatures: Dict[str, str] = field(default_factory=dict)  # backend → normalized sig
    n_valid_comparisons: int = 0   # backends that produced valid numerical s_diff
    # ── Bug evidence (set when a real bug is found) ───────────────────────────
    # backend → flat float32 list (for JSON serialisation)
    bug_inputs: Dict[str, List] = field(default_factory=dict)      # input_name → values
    expected_outputs: Dict[str, List] = field(default_factory=dict) # backend → noopt output
    buggy_outputs: Dict[str, List] = field(default_factory=dict)    # backend → opt output
    noopt_valid: Dict[str, bool] = field(default_factory=dict)      # backend → noopt≈ref
    # Consensus-based reference correction (set by _consensus_check):
    # When ≥3 target backends agree pairwise but disagree with the reference,
    # the reference is the likely bug — every per-backend s_diff is zeroed
    # and this flag is set so callers can skip the candidate.
    reference_likely_wrong: bool = False
    consensus_size: int = 0   # number of targets that agreed in the consensus group


class DiscrepancyOracle:
    """
    Manages all backends and computes the unified discrepancy score S(m).

    Crashes are recorded in crash_info but do NOT contribute to total_score.
    Only genuine numerical divergence between pytorch_eager and a target
    (or between opt/no-opt runs of the same target) affects the score.
    """

    _BACKEND_CLASSES = {
        "pytorch_eager":  PyTorchEagerBackend,
        "onnxruntime":    ONNXRuntimeBackend,
        "tvm":            TVMBackend,
        "tensorrt":       TensorRTBackend,
        "xla":            XLABackend,
        "torchscript":    TorchScriptBackend,
        "torch_compile":  TorchCompileBackend,
        "tensorflow":     TFBackend,
        "tflite":         TFLiteBackend,
        "openvino":       OpenVINOBackend,
    }

    def __init__(self, config: TrionConfig) -> None:
        self.config = config
        self.delta  = config.tolerance

        self._ref_backend:    Optional[BackendBase] = None
        self._target_backends: List[BackendBase]    = []

        self._init_backends()

    def _init_backends(self) -> None:
        cfg = self.config

        # Reference
        ref_cls = self._BACKEND_CLASSES.get(cfg.reference_backend)
        if ref_cls:
            b = ref_cls()
            if b.is_available():
                self._ref_backend = b
            else:
                logger.warning("Reference backend '%s' not available.",
                               cfg.reference_backend)

        # Targets
        for bname in cfg.target_backends:
            cls = self._BACKEND_CLASSES.get(bname)
            if cls is None:
                logger.warning("Unknown backend: %s", bname)
                continue
            kwargs: dict = {}
            if bname == "tvm":
                kwargs = {"target": cfg.tvm_target, "opt_level": cfg.tvm_opt_level}
            elif bname == "tensorrt":
                kwargs = {"fp16": cfg.tensorrt_fp16}
            b = cls(**kwargs)
            if b.is_available():
                self._target_backends.append(b)
            else:
                logger.info("Backend '%s' not available (skipped).", bname)

        if not self._target_backends:
            logger.warning("No target backends available; oracle will return 0.")

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        model_id: int = 0,
    ) -> OracleReport:
        report = OracleReport(model_id=model_id, total_score=0.0)

        # Store input arrays once (shared across all backends)
        report.bug_inputs = {
            k: v.flatten().tolist() for k, v in inputs.items()
        }

        # Run reference (pytorch_eager, no optimization)
        ref_result: Optional[BackendResult] = None
        if self._ref_backend:
            ref_result = self._ref_backend.run(model, inputs, optimized=False)
            if ref_result.crashed:
                label = "ref:" + self._ref_backend.name
                report.crashes.append(label)
                err = ref_result.error or ""
                report.errors[label] = err
                report.crash_info[label] = _classify_crash(err)
                ref_result = None   # treat as unavailable; s_diff will be 0
            elif ref_result.output is not None:
                # Discard models where the reference itself is numerically
                # invalid — NaN/Inf in the reference means we have no ground
                # truth to compare against, so any discrepancy is meaningless.
                if not np.all(np.isfinite(ref_result.output)):
                    logger.debug(
                        "Model %d: reference output contains NaN/Inf — skipping.",
                        model_id,
                    )
                    return report  # total_score stays 0; not a real bug

        # Collect every target's noopt output before scoring so the
        # consensus check can decide whether the reference is wrong.
        per_backend_runs: Dict[str, dict] = {}
        for backend in self._target_backends:
            res_opt   = backend.run(model, inputs, optimized=True)
            res_noopt = backend.run(model, inputs, optimized=False)
            per_backend_runs[backend.name] = {"opt": res_opt, "noopt": res_noopt}

        # If ≥3 targets pairwise agree and the reference disagrees, treat the
        # reference as the buggy party — clear it and score all targets as 0.
        ref_was_wrong = self._reference_disagrees_with_consensus(
            ref_result, per_backend_runs, report,
        )
        if ref_was_wrong:
            ref_result = None  # downstream s_diff falls back to 0.0

        for backend in self._target_backends:
            bname = backend.name
            res_opt   = per_backend_runs[bname]["opt"]
            res_noopt = per_backend_runs[bname]["noopt"]

            # Record crashes — but do NOT count them toward total_score
            if res_opt.crashed:
                label = f"{bname}+opt"
                report.crashes.append(label)
                err = res_opt.error or ""
                report.errors[label] = err
                report.crash_info[label] = _classify_crash(err)
                report.error_signatures[label] = _error_signature(err)

            if res_noopt.crashed:
                label = f"{bname}-opt"
                report.crashes.append(label)
                err = res_noopt.error or ""
                report.errors[label] = err
                report.crash_info[label] = _classify_crash(err)
                report.error_signatures[label] = _error_signature(err)

            # Validate noopt ≈ pytorch_eager reference.
            # Only count Δ_opt as a real bug if the baseline (noopt) is correct.
            noopt_ok = self._noopt_matches_ref(ref_result, res_noopt)
            report.noopt_valid[bname] = noopt_ok

            # S_diff: cross-backend inconsistency (includes status-differ = 1.0)
            s_diff = self._compute_s_diff(ref_result, res_opt, bname)
            report.s_diff[bname] = s_diff
            # Count as valid comparison only when both produced numerical output
            if (ref_result is not None and ref_result.output is not None
                    and not res_opt.crashed and res_opt.output is not None):
                report.n_valid_comparisons += 1

            # Δ_opt — only count when noopt itself is correct (noopt ≈ ref)
            # If noopt is already wrong, the discrepancy may not be opt-induced.
            if noopt_ok:
                delta_opt = self._compute_delta_opt(res_opt, res_noopt)
            else:
                delta_opt = 0.0
            report.delta_opt[bname] = delta_opt

            report.total_score += s_diff + delta_opt

            # Always record per-backend evidence (noopt output as "expected",
            # opt output as "buggy") and the PyTorch reference, so that
            # downstream reproducer scripts have all three sides of the
            # comparison even when the bug is borderline.
            if res_noopt.output is not None:
                report.expected_outputs[bname] = res_noopt.output.flatten().tolist()
            if res_opt.output is not None:
                report.buggy_outputs[bname] = res_opt.output.flatten().tolist()

        if (ref_result is not None
                and ref_result.output is not None):
            report.expected_outputs["__reference__"] = (
                ref_result.output.flatten().tolist()
            )

        return report

    def _reference_disagrees_with_consensus(
        self,
        ref: Optional[BackendResult],
        per_backend_runs: Dict[str, dict],
        report: "OracleReport",
    ) -> bool:
        """Detect a wrong reference using cross-target consensus.

        Heuristic: if at least 3 target backends produced numerical output
        with the *same shape* and they pairwise agree (rel_L2 ≤ self.delta),
        but the reference's output diverges from them by more than self.delta,
        then the reference is the bug — independent compilers very rarely
        produce the same wrong answer for the same model.

        Marks `report.reference_likely_wrong = True` and records the
        consensus group size so callers can skip the candidate.
        """
        if ref is None or ref.output is None:
            return False
        try:
            ref_flat = ref.output.flatten().astype(np.float64)
        except Exception:
            return False

        candidates = []  # list of (name, flat float64 vector)
        for name, runs in per_backend_runs.items():
            r = runs["noopt"]
            if r.crashed or r.output is None:
                continue
            try:
                v = r.output.flatten().astype(np.float64)
            except Exception:
                continue
            if v.shape != ref_flat.shape:
                continue
            candidates.append((name, v))

        if len(candidates) < 3:
            return False

        def _close(a: np.ndarray, b: np.ndarray) -> bool:
            denom = max(float(np.linalg.norm(b)), 1e-3)
            return float(np.linalg.norm(a - b) / denom) <= self.delta

        # Find the largest set of mutually-agreeing targets.
        n = len(candidates)
        best_group: List[int] = []
        for i in range(n):
            group = [i]
            for j in range(n):
                if j == i:
                    continue
                if all(_close(candidates[j][1], candidates[k][1]) for k in group):
                    group.append(j)
            if len(group) > len(best_group):
                best_group = group

        if len(best_group) < 3:
            return False

        # Reference must disagree with the consensus group.
        consensus_vec = candidates[best_group[0]][1]
        if _close(consensus_vec, ref_flat):
            return False

        report.reference_likely_wrong = True
        report.consensus_size = len(best_group)
        logger.info(
            "Model %d: reference disagrees with %d-target consensus — "
            "treating reference as buggy, no per-target bug recorded.",
            report.model_id, len(best_group),
        )
        return True

    def _noopt_matches_ref(
        self,
        ref: Optional[BackendResult],
        noopt: BackendResult,
    ) -> bool:
        """Return True if noopt output is close enough to the pytorch_eager reference.
        When ref is unavailable (crashed), assume noopt is valid (can't check)."""
        if ref is None or ref.output is None:
            return True   # cannot validate; give benefit of the doubt
        if noopt.crashed or noopt.output is None:
            return False
        try:
            r = ref.output.flatten().astype(np.float64)
            n = noopt.output.flatten().astype(np.float64)
            # Shape mismatch is a real divergence, not a "match".
            if r.shape != n.shape:
                return False
            diff = np.linalg.norm(r - n)
            norm = np.linalg.norm(r) + 1e-8
            return float(diff / norm) <= (self.delta * 10)  # 10× looser for noopt check
        except Exception:
            return True

    # ── Score components ──────────────────────────────────────────────────────

    def _compute_s_diff(
        self,
        ref: Optional[BackendResult],
        tar: BackendResult,
        bname: str,
    ) -> float:
        """Cross-backend inconsistency S_diff(m) per the paper oracle.

        S_diff = 1                              if execution status differs
               = min(1, rel_diff / δ)           otherwise (both ran)
               = 0                              if ref unavailable

        "Execution status differs" means exactly one side produced output;
        the other crashed.  Crashes are also logged separately in crash_info.
        """
        ref_ok = (ref is not None and not ref.crashed
                  and ref.output is not None)
        tar_ok = (not tar.crashed and tar.output is not None)

        # Ref unavailable → can't compute meaningful S_diff
        if not ref_ok:
            return 0.0

        # Execution status differs: ref ran, target crashed → S_diff = 1
        if not tar_ok:
            return 1.0

        # Both ran: compute relative L2 divergence
        try:
            r = ref.output.flatten().astype(np.float64)
            t = tar.output.flatten().astype(np.float64)
            # Shape mismatch ⇒ structural divergence: cap at 1.0.
            if r.shape != t.shape:
                return 1.0
            diff = np.linalg.norm(r - t)
            # Use max(||ref||, abs_floor) as denominator so that near-zero
            # reference outputs don't inflate the relative error into false
            # positives.  1e-3 means: if the reference output norm is smaller
            # than 1e-3 we treat it as if it were 1e-3 for the purposes of
            # the ratio.  A real divergence on a near-zero output would need
            # to exceed delta * 1e-3 in absolute terms to fire.
            _ABS_FLOOR = 1e-3
            norm = max(float(np.linalg.norm(r)), _ABS_FLOOR)
            raw  = (diff / norm) / self.delta
            return float(min(1.0, raw)) if np.isfinite(raw) else 0.0
        except Exception as exc:
            logger.debug("S_diff computation error (%s): %s", bname, exc)
            return 0.0

    def _compute_delta_opt(
        self,
        res_opt:   BackendResult,
        res_noopt: BackendResult,
    ) -> float:
        """Optimization-induced deviation Δ_opt(m) per the paper oracle.

        Δ_opt = min(1, ||y_tar+ - y_tar-||_rel / δ)

        Returns 0.0 if either run crashed (crashes are logged separately).
        """
        if res_opt.crashed or res_noopt.crashed:
            return 0.0
        if res_opt.output is None or res_noopt.output is None:
            return 0.0
        try:
            a = res_opt.output.flatten().astype(np.float64)
            b = res_noopt.output.flatten().astype(np.float64)
            # Shape mismatch between opt and noopt is itself a real divergence.
            if a.shape != b.shape:
                return 1.0
            diff = np.linalg.norm(a - b)
            _ABS_FLOOR = 1e-3
            norm = max(float(np.linalg.norm(b)), _ABS_FLOOR)
            raw  = (diff / norm) / self.delta
            return float(min(1.0, raw)) if np.isfinite(raw) else 0.0
        except Exception as exc:
            logger.debug("Δ_opt computation error: %s", exc)
            return 0.0

    # ── Availability summary ──────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["=== Backend Availability ==="]
        ref_name = self._ref_backend.name if self._ref_backend else "NONE"
        lines.append(f"  Reference : {ref_name}")
        lines.append(f"  Targets   : {[b.name for b in self._target_backends]}")
        return "\n".join(lines)
