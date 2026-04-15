"""
Cross-backend Oracle (Section 2.5, redesigned 2026-04-15).

The oracle reports two independent bug signals:

  Oracle 1 — Cross-backend inconsistency.
      S(m) = max_{i ≠ j}  min( 1, ||y_i − y_j|| / δ )
    over all *target* backends that produced numerical output for the
    same model m and input x.  No single reference is trusted: any
    pairwise disagreement above tolerance is the signal.  This avoids
    the "reference is wrong" failure mode that produced ~50 historical
    false positives when the harness's reference (onnx2torch) was
    silently buggy.

  Oracle 2 — Crash.
      Recorded per-backend in `report.crashes` and never folded into
      total_score.  A target that crashed contributes neither to
      Oracle 1 nor to its own crash entry's score; downstream filtering
      handles "frontend" vs "backend" classification.

Eager helpers (PyTorch eager, JAX eager, TF eager) are NOT used to
compute S(m).  They are consulted *post hoc* — when Oracle 1 fires —
to attribute blame: the target whose output is farthest from the
eager consensus is named in `report.suspect_backend`, but the score
itself is independent of any eager run.

Backend availability fallbacks:

  - If the user requests `xla` (JAX→XLA) and JAX is missing, the
    oracle silently registers `tensorflow` (TF→XLA via
    tf.function(jit_compile=True)) under the same logical role so XLA
    coverage is preserved.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    "onnx2torch",
    "FakeTensor", "fake_tensor",
    "make_fx", "symbolic_trace", "torch.fx",
    "dynamo",
    "BackendCompilerFailed",
]
_BACKEND_KEYWORDS = [
    "CUDA", "cudnn",
    "RuntimeError",
    "illegal memory",
    "INTERNAL ASSERT",
    "lax.", "conv_general",
    "shape mismatch",
    "DType",
    "Not implemented",
]


# ── Shared-infrastructure pairs ───────────────────────────────────────────────
#
# Pairs of backends that share enough plumbing that a divergence of just that
# pair from the rest of the field is more likely to be a harness bug (in the
# shared layer) than two independent compiler bugs that happen to agree.
# When the worst-pair under Oracle 1 falls inside one of these groups AND
# the rest of the field agrees, the report's `shared_infra_warning` flag is
# set so triage can investigate the shared layer first.
SHARED_INFRA: Dict[frozenset, str] = {
    # xla and tflite both consume the model through trion.oracle._onnx_ops.py
    # dispatch (jax.numpy and tf.experimental.numpy paths). A bug in the
    # dispatcher fires on both — looks like consensus, isn't.
    frozenset({"xla", "tflite"}): "_onnx_ops dispatch (jnp + tf.experimental)",
    # torch.compile and torchscript both go through onnx2torch. onnx2torch
    # is the single biggest historical false-positive source.
    frozenset({"torch_compile", "torchscript"}): "onnx2torch converter",
    # tensorflow (TFBackend, jit_compile=True) and xla both ultimately run
    # XLA HLO. A real XLA bug WILL fire on both — but so will a harness
    # tf_backend bug. Treat as ambiguous.
    frozenset({"xla", "tensorflow"}): "XLA HLO (shared codegen)",
}


# ── Frontend-vulnerable ops ───────────────────────────────────────────────────
#
# When a model contains any of these op types, the listed backend's reading
# path (onnx2torch in particular) is known to silently mis-convert it. The
# oracle excludes those backends for that specific model so the divergence
# they produce isn't attributed to the compiler.
#
# Format: backend name → set of ONNX op_types that the backend's frontend
# (not the compiler itself) is documented to mishandle.
FRONTEND_VULNERABLE_OPS: Dict[str, set] = {
    "torch_compile": {
        # onnx2torch CumSum converter calls axis.item() → graph break.
        "CumSum",
    },
    "torchscript": {
        # Same onnx2torch issues affect TorchScript path.
        "CumSum",
    },
}


def _classify_crash(error: str) -> str:
    if any(k in error for k in _FRONTEND_KEYWORDS):
        return "frontend"
    return "backend"


def _error_signature(error: str) -> str:
    import re
    first_line = next((l.strip() for l in error.splitlines() if l.strip()), error)
    sig = re.sub(r"0x[0-9a-fA-F]+", "<addr>", first_line)
    sig = re.sub(r"\b\d+\b", "N", sig)
    return sig[:120]


@dataclass
class OracleReport:
    """Per-model oracle result."""
    model_id: int
    total_score: float                                            # = Oracle 1, max pairwise

    # Oracle 1: cross-backend pairwise divergence
    pairwise_divergence: Dict[Tuple[str, str], float] = field(default_factory=dict)
    worst_pair: Optional[Tuple[str, str]] = None                  # (b_i, b_j) at the max
    suspect_backend: Optional[str] = None                         # name attributed by eager helper
    suspect_distance_to_eager: float = 0.0
    eager_helper_used: bool = False                               # True iff eager ran successfully
    n_valid_targets: int = 0                                      # targets that produced output

    # Oracle 2: crash channel (independent)
    crashes: List[str] = field(default_factory=list)              # crashed backend labels
    errors: Dict[str, str] = field(default_factory=dict)          # backend → error msg
    crash_info: Dict[str, str] = field(default_factory=dict)      # backend → "frontend"|"backend"
    error_signatures: Dict[str, str] = field(default_factory=dict)

    # Bug evidence (always populated when total_score > 0)
    bug_inputs: Dict[str, List] = field(default_factory=dict)     # input_name → flat list
    target_outputs: Dict[str, List] = field(default_factory=dict) # backend → flat list
    eager_output: Optional[List] = None                           # eager helper's output, if any

    # Triage hints (Oracle 1 + post-hoc heuristics)
    shared_infra_warning: Optional[str] = None  # "_onnx_ops dispatch", etc.
    skipped_targets: Dict[str, str] = field(default_factory=dict)
    # ↑ backend name → reason (e.g. "torch_compile: model contains CumSum
    # which onnx2torch mishandles"). These targets did NOT participate in the
    # pairwise score; counted under crashes nor under valid targets.

    # Back-compat fields for downstream code that read the old report.
    s_diff: Dict[str, float] = field(default_factory=dict)
    delta_opt: Dict[str, float] = field(default_factory=dict)
    expected_outputs: Dict[str, List] = field(default_factory=dict)
    buggy_outputs: Dict[str, List] = field(default_factory=dict)
    noopt_valid: Dict[str, bool] = field(default_factory=dict)
    n_valid_comparisons: int = 0
    reference_likely_wrong: bool = False
    consensus_size: int = 0


class DiscrepancyOracle:
    """Cross-backend pairwise oracle (Oracle 1) + crash channel (Oracle 2)."""

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

        self._eager_backend: Optional[BackendBase] = None  # blame helper only
        self._target_backends: List[BackendBase] = []
        self._init_backends()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_backends(self) -> None:
        cfg = self.config

        # Eager helper (used only post-hoc to attribute blame).
        ref_cls = self._BACKEND_CLASSES.get(cfg.reference_backend)
        if ref_cls:
            b = ref_cls()
            if b.is_available():
                self._eager_backend = b
            else:
                logger.warning("Eager helper '%s' not available — Oracle 1 "
                               "will still fire but blame attribution will be "
                               "skipped.", cfg.reference_backend)

        # Targets — with XLA → TF-XLA fallback.
        seen_names: set = set()
        for bname in cfg.target_backends:
            backend = self._instantiate_target(bname)
            if backend is None:
                continue
            # Avoid registering the same backend twice when xla→tensorflow
            # fallback collides with an explicit `tensorflow` entry.
            if backend.name in seen_names:
                logger.info("Backend '%s' already registered — skipping "
                            "duplicate from '%s'.", backend.name, bname)
                continue
            self._target_backends.append(backend)
            seen_names.add(backend.name)

        if not self._target_backends:
            logger.warning("No target backends available; oracle will return 0.")
        elif len(self._target_backends) == 1:
            logger.warning("Only 1 target backend available — Oracle 1 needs "
                           "≥ 2 to compute any pairwise divergence.")

    def _instantiate_target(self, bname: str) -> Optional[BackendBase]:
        """Build a backend instance, falling back from xla → tensorflow when
        JAX is missing so XLA coverage is preserved."""
        cfg = self.config
        cls = self._BACKEND_CLASSES.get(bname)
        if cls is None:
            logger.warning("Unknown backend: %s", bname)
            return None
        kwargs: dict = {}
        if bname == "tvm":
            kwargs = {"target": cfg.tvm_target, "opt_level": cfg.tvm_opt_level}
        elif bname == "tensorrt":
            kwargs = {"fp16": cfg.tensorrt_fp16}
        b = cls(**kwargs)
        if b.is_available():
            return b
        if bname == "xla":
            # XLA via JAX unavailable — fall back to TF-XLA so we still test
            # XLA's algebraic simplifier / HLO passes.
            tf_b = self._BACKEND_CLASSES["tensorflow"]()
            if tf_b.is_available():
                logger.info("xla backend not available; substituting "
                            "tensorflow (TF-XLA via jit_compile=True) so XLA "
                            "coverage is preserved.")
                return tf_b
        logger.info("Backend '%s' not available (skipped).", bname)
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        model_id: int = 0,
    ) -> OracleReport:
        report = OracleReport(model_id=model_id, total_score=0.0)
        report.bug_inputs = {k: v.flatten().tolist() for k, v in inputs.items()}

        # Pre-flight: identify which backends should be skipped for this
        # model because their *frontend* (not their compiler) is known to
        # mishandle one of the model's ops. Without this filter, every
        # CumSum-containing model would falsely flag torch.compile and
        # TorchScript as buggy, when the bug is actually in onnx2torch.
        op_types = {n.op_type for n in model.graph.node}
        skip = self._skipped_targets_for(op_types)
        report.skipped_targets = skip

        # Run every target with optimisation enabled. The new oracle has no
        # noopt comparison — that was the Δ_opt term of the prior design,
        # which the new Oracle 1 spec doesn't include.
        target_runs: Dict[str, BackendResult] = {}
        for backend in self._target_backends:
            if backend.name in skip:
                continue   # frontend-vulnerable for this op set
            res = backend.run(model, inputs, optimized=True)
            self._record_crash_if_any(report, backend.name, res)
            target_runs[backend.name] = res

        # Oracle 1: max pairwise divergence over targets that produced output.
        valid: List[Tuple[str, np.ndarray]] = []
        for name, res in target_runs.items():
            if res.crashed or res.output is None:
                continue
            try:
                arr = np.asarray(res.output).astype(np.float64)
            except Exception:
                continue
            if not np.all(np.isfinite(arr)):
                # NaN / Inf in a target output is a bug, but it isn't a
                # divergence we can score numerically — record under the
                # crash channel as "backend" and skip.
                report.crashes.append(f"{name}+nan")
                report.errors[f"{name}+nan"] = "non-finite output"
                report.crash_info[f"{name}+nan"] = "backend"
                continue
            valid.append((name, arr))
            report.target_outputs[name] = arr.flatten().tolist()
        report.n_valid_targets = len(valid)

        if len(valid) >= 2:
            report.total_score = self._oracle1_pairwise(valid, report)

        # Oracle 1 + eager blame helper: only when divergence is real.
        if report.total_score > 0.0 and self._eager_backend is not None:
            self._attribute_blame_via_eager(model, inputs, valid, report)

        # Triage: if the worst-pair is a known shared-infra pair AND the
        # rest of the field agrees with each other, raise a warning so the
        # user investigates the shared layer (dispatch / converter / shared
        # codegen) before filing a bug against either compiler.
        if report.worst_pair is not None and report.total_score > 0.0:
            warning = self._maybe_shared_infra_warning(report, valid)
            if warning is not None:
                report.shared_infra_warning = warning

        # Extra triage: if ≥2 members of any SHARED_INFRA group CRASHED
        # (rather than merely agreeing on a wrong value), the surviving
        # targets may include torch_compile-via-onnx2torch, whose converter
        # can silently mis-translate the very ops that made the dispatch
        # group crash. Flag this pattern so downstream classification
        # treats the score as untrusted. Example from the 500-model run:
        # 9/14 "bugs" were models with ReduceL1 → xla/tflite/tvm all fail
        # "Unsupported ONNX op: ReduceL1" in _onnx_ops dispatch, leaving
        # openvino/onnxruntime/torch_compile, where torch_compile produced
        # wrong output via onnx2torch's ReduceL1 converter — the score was
        # NOT a real Inductor bug.
        if report.total_score > 0.0 and report.shared_infra_warning is None:
            crashed = {
                label.rsplit("+opt", 1)[0]
                for label in report.crashes
                if label.endswith("+opt")
            }
            for group, infra_label in SHARED_INFRA.items():
                group_crashed = [g for g in group if g in crashed]
                if len(group_crashed) >= 2:
                    report.shared_infra_warning = (
                        f"{sorted(group_crashed)} (sharing {infra_label}) "
                        "all crashed on this model; score computed on the "
                        "surviving targets may reflect the converter of "
                        "whichever surviving backend depends on onnx2torch "
                        "or a similar bridge. Inspect the error messages "
                        "in report.errors before filing a compiler bug."
                    )
                    break

        # For back-compat with downstream code that expected the old fields.
        report.s_diff = {name: report.total_score for name, _ in valid}
        if valid:
            report.expected_outputs["__cross_max__"] = (
                valid[0][1].flatten().tolist()
            )

        return report

    # ── Oracle 1: pairwise divergence ────────────────────────────────────────

    def _oracle1_pairwise(
        self,
        valid: List[Tuple[str, np.ndarray]],
        report: OracleReport,
    ) -> float:
        """S(m) = max_{i≠j} min(1, ||y_i − y_j|| / δ).

        Records every pairwise divergence in `report.pairwise_divergence` and
        the worst pair in `report.worst_pair`.
        """
        max_div = 0.0
        worst: Optional[Tuple[str, str]] = None
        for i in range(len(valid)):
            ni, yi = valid[i]
            for j in range(i + 1, len(valid)):
                nj, yj = valid[j]
                if yi.shape != yj.shape:
                    div = 1.0  # structural divergence — capped
                else:
                    diff = float(np.linalg.norm(yi.flatten() - yj.flatten()))
                    norm = max(
                        float(np.linalg.norm(yi.flatten())),
                        float(np.linalg.norm(yj.flatten())),
                        1e-3,
                    )
                    raw = (diff / norm) / self.delta
                    div = float(min(1.0, raw)) if np.isfinite(raw) else 0.0
                pair_key = (ni, nj)
                report.pairwise_divergence[pair_key] = div
                if div > max_div:
                    max_div = div
                    worst = pair_key
        report.worst_pair = worst
        return max_div

    # ── Eager blame attribution (post-hoc only) ──────────────────────────────

    def _attribute_blame_via_eager(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        valid: List[Tuple[str, np.ndarray]],
        report: OracleReport,
    ) -> None:
        """Run the eager helper once; blame the target farthest from it.

        This does NOT change Oracle 1's score. It only annotates the report
        so a downstream triage tool can suggest which compiler to look at
        first. If eager itself crashes or produces non-finite output, no
        blame is attributed — the divergence is still reported, just
        without a suspect.
        """
        try:
            res = self._eager_backend.run(model, inputs, optimized=False)
        except Exception as exc:
            logger.debug("eager helper crashed: %s", exc)
            return
        if res.crashed or res.output is None:
            return
        try:
            ref = np.asarray(res.output).astype(np.float64)
        except Exception:
            return
        if not np.all(np.isfinite(ref)):
            return
        report.eager_helper_used = True
        report.eager_output = ref.flatten().tolist()
        # Distance per target — only those with matching shape can be ranked.
        distances: Dict[str, float] = {}
        ref_flat = ref.flatten()
        ref_norm = max(float(np.linalg.norm(ref_flat)), 1e-3)
        for name, arr in valid:
            arr_flat = arr.flatten()
            if arr_flat.shape != ref_flat.shape:
                # Shape mismatch with eager — that target is structurally
                # wrong; max possible distance.
                distances[name] = float("inf")
                continue
            distances[name] = float(np.linalg.norm(arr_flat - ref_flat) / ref_norm)
        if not distances:
            return
        suspect = max(distances, key=distances.get)
        report.suspect_backend = suspect
        report.suspect_distance_to_eager = distances[suspect]

    # ── False-positive guards ────────────────────────────────────────────────

    def _skipped_targets_for(self, op_types: set) -> Dict[str, str]:
        """Return backend → reason for every backend that should be skipped
        on a model containing these op_types because the backend's *frontend*
        (not its compiler) is documented to mishandle them.

        This is the cheapest of the three FP defences: it never even runs
        the backend in question, so there's nothing to score and nothing
        to crash. Only backends explicitly listed in FRONTEND_VULNERABLE_OPS
        are affected.
        """
        skipped: Dict[str, str] = {}
        for be in self._target_backends:
            bad = FRONTEND_VULNERABLE_OPS.get(be.name)
            if not bad:
                continue
            hits = bad & op_types
            if hits:
                skipped[be.name] = (
                    f"frontend mishandles {sorted(hits)} for this backend"
                )
        return skipped

    def _maybe_shared_infra_warning(
        self,
        report: OracleReport,
        valid: List[Tuple[str, np.ndarray]],
    ) -> Optional[str]:
        """Detect the "shared-infrastructure outlier" pattern.

        Pattern: two backends from a SHARED_INFRA group both produce the
        SAME (wrong) output, while every backend outside that group
        produces a DIFFERENT but mutually consistent answer. This is the
        signature of a bug in the shared layer (dispatch / converter /
        shared codegen), not of two simultaneously-buggy compilers.

        Crucially we do NOT just look at the worst pair — when xla and
        tflite both compute the same wrong value, the worst pair is
        actually e.g. (xla, onnxruntime), not (xla, tflite). So we
        enumerate every shared-infra group whose members are all
        present, then check the within-group / to-rest / others-internal
        divergence triple.
        """
        if len(valid) < 3:
            return None
        out_map = {n: a for n, a in valid}
        delta = self.delta

        def _div(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape != b.shape:
                return 1.0
            af = a.flatten(); bf = b.flatten()
            norm = max(float(np.linalg.norm(af)),
                       float(np.linalg.norm(bf)), 1e-3)
            return min(1.0, (float(np.linalg.norm(af - bf)) / norm) / delta)

        for group, infra_label in SHARED_INFRA.items():
            members = [n for n in group if n in out_map]
            if len(members) < 2:
                continue
            others = [(n, a) for n, a in valid if n not in group]
            if len(others) < 2:
                continue
            # Within-group: members must agree with each other.
            within = max(_div(out_map[a], out_map[b])
                         for a in members for b in members if a < b)
            if within > 0.5:
                continue
            # Members must collectively disagree with the others.
            to_rest = max(_div(out_map[m], a)
                          for m in members for _, a in others)
            if to_rest <= 0.5:
                continue
            # Others must agree with each other.
            others_internal = max(
                _div(others[i][1], others[j][1])
                for i in range(len(others))
                for j in range(i + 1, len(others))
            )
            if others_internal > 0.5:
                continue
            # Pattern matches: shared-infra is the suspect.
            return (
                f"{', '.join(sorted(members))} share {infra_label} and "
                f"agree on a value that diverges from the remaining "
                f"{len(others)} backend(s) "
                f"({', '.join(sorted(n for n, _ in others))}); "
                "investigate the shared layer before filing a compiler bug."
            )
        return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _record_crash_if_any(
        self,
        report: OracleReport,
        bname: str,
        res: BackendResult,
    ) -> None:
        if not res.crashed:
            return
        label = f"{bname}+opt"
        report.crashes.append(label)
        err = res.error or ""
        report.errors[label] = err
        report.crash_info[label] = _classify_crash(err)
        report.error_signatures[label] = _error_signature(err)

    # ── Availability summary ─────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["=== Backend Availability ==="]
        eager_name = self._eager_backend.name if self._eager_backend else "NONE"
        lines.append(f"  Eager (blame helper) : {eager_name}")
        lines.append(f"  Targets ({len(self._target_backends)}) : "
                     f"{[b.name for b in self._target_backends]}")
        lines.append(f"  Tolerance δ           : {self.delta}")
        return "\n".join(lines)
