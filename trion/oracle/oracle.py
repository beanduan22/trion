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

        for backend in self._target_backends:
            bname = backend.name

            # Optimized run (y_tar+)
            res_opt   = backend.run(model, inputs, optimized=True)
            # Unoptimized run (y_tar-)
            res_noopt = backend.run(model, inputs, optimized=False)

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

            # S_diff — only when both ref and target (opt) ran successfully
            s_diff = self._compute_s_diff(ref_result, res_opt, bname)
            report.s_diff[bname] = s_diff
            if ref_result is not None and not res_opt.crashed:
                report.n_valid_comparisons += 1

            # Δ_opt — only when both opt/noopt ran successfully
            delta_opt = self._compute_delta_opt(res_opt, res_noopt)
            report.delta_opt[bname] = delta_opt

            report.total_score += s_diff + delta_opt

        return report

    # ── Score components ──────────────────────────────────────────────────────

    def _compute_s_diff(
        self,
        ref: Optional[BackendResult],
        tar: BackendResult,
        bname: str,
    ) -> float:
        """Numerical divergence between reference output and target output.
        Returns 0.0 if either side crashed — crashes are logged separately.
        """
        if ref is None or ref.crashed or tar.crashed:
            return 0.0
        if ref.output is None or tar.output is None:
            return 0.0
        try:
            r = ref.output.flatten().astype(np.float32)
            t = tar.output.flatten().astype(np.float32)
            n = min(len(r), len(t))
            diff  = np.linalg.norm((r[:n] - t[:n]).astype(np.float64))
            norm  = np.linalg.norm(r[:n].astype(np.float64))
            denom = max(norm * self.delta, self.delta * 10) + 1e-8
            raw   = diff / denom
            return float(min(1.0, raw)) if np.isfinite(raw) else 0.0
        except Exception as exc:
            logger.debug("S_diff computation error (%s): %s", bname, exc)
            return 0.0

    def _compute_delta_opt(
        self,
        res_opt:   BackendResult,
        res_noopt: BackendResult,
    ) -> float:
        """Optimization-induced numerical deviation within the same backend.
        Returns 0.0 if either run crashed — crashes are logged separately.
        """
        if res_opt.crashed or res_noopt.crashed:
            return 0.0
        if res_opt.output is None or res_noopt.output is None:
            return 0.0
        try:
            a = res_opt.output.flatten().astype(np.float32)
            b = res_noopt.output.flatten().astype(np.float32)
            n = min(len(a), len(b))
            diff  = np.linalg.norm((a[:n] - b[:n]).astype(np.float64))
            norm  = np.linalg.norm(b[:n].astype(np.float64))
            denom = max(norm * self.delta, self.delta * 10) + 1e-8
            raw   = diff / denom
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
