"""
Optimization-Sensitive Oracles (Section 2.5).

S(m) = S_diff(m) + Δ_opt(m)

S_diff:   cross-backend inconsistency   (ref vs target)
Δ_opt:    optimization-induced deviation (target+opt vs target−opt)
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

logger = logging.getLogger(__name__)


@dataclass
class OracleReport:
    """Per-model oracle result."""
    model_id: int
    total_score: float
    s_diff: Dict[str, float] = field(default_factory=dict)   # backend → score
    delta_opt: Dict[str, float] = field(default_factory=dict) # backend → score
    crashes: List[str] = field(default_factory=list)
    errors:  Dict[str, str] = field(default_factory=dict)


class DiscrepancyOracle:
    """
    Manages all backends and computes the unified discrepancy score S(m).
    """

    _BACKEND_CLASSES = {
        "pytorch_eager": PyTorchEagerBackend,
        "onnxruntime":   ONNXRuntimeBackend,
        "tvm":           TVMBackend,
        "tensorrt":      TensorRTBackend,
        "xla":           XLABackend,
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

        # Run reference
        ref_result: Optional[BackendResult] = None
        if self._ref_backend:
            ref_result = self._ref_backend.run(model, inputs, optimized=False)
            if ref_result.crashed:
                report.crashes.append(self._ref_backend.name)
                report.errors[self._ref_backend.name] = ref_result.error or ""

        for backend in self._target_backends:
            bname = backend.name

            # Optimized run (y_tar+)
            res_opt  = backend.run(model, inputs, optimized=True)
            # Unoptimized run (y_tar-)
            res_noopt = backend.run(model, inputs, optimized=False)

            # Cross-backend inconsistency S_diff
            s_diff = self._compute_s_diff(ref_result, res_opt, bname, report)
            report.s_diff[bname] = s_diff

            # Optimization-induced deviation Δ_opt
            delta_opt = self._compute_delta_opt(res_opt, res_noopt)
            report.delta_opt[bname] = delta_opt

            report.total_score += s_diff + delta_opt

            if res_opt.crashed:
                report.crashes.append(f"{bname}+opt")
            if res_noopt.crashed:
                report.crashes.append(f"{bname}-opt")
            if res_opt.error:
                report.errors[f"{bname}+opt"] = res_opt.error
            if res_noopt.error:
                report.errors[f"{bname}-opt"] = res_noopt.error

        return report

    # ── Score components ──────────────────────────────────────────────────────

    def _compute_s_diff(
        self,
        ref: Optional[BackendResult],
        tar: BackendResult,
        bname: str,
        report: OracleReport,
    ) -> float:
        # Execution status mismatch
        if ref is None:
            return 0.0
        if ref.crashed != tar.crashed:
            return 1.0
        if ref.crashed and tar.crashed:
            return 0.0
        if ref.output is None or tar.output is None:
            return 0.0
        try:
            r = ref.output.flatten().astype(np.float32)
            t = tar.output.flatten().astype(np.float32)
            # Align shapes (take min length)
            n = min(len(r), len(t))
            diff = np.linalg.norm((r[:n] - t[:n]).astype(np.float64))
            norm = max(np.linalg.norm(r[:n].astype(np.float64)), 1e-8)
            raw  = diff / (norm * self.delta + 1e-8)
            return float(min(1.0, raw)) if np.isfinite(raw) else 0.0
        except Exception as exc:
            logger.debug("S_diff computation error (%s): %s", bname, exc)
            return 0.0

    def _compute_delta_opt(
        self,
        res_opt:   BackendResult,
        res_noopt: BackendResult,
    ) -> float:
        if res_opt.crashed or res_noopt.crashed:
            return float(res_opt.crashed != res_noopt.crashed)
        if res_opt.output is None or res_noopt.output is None:
            return 0.0
        try:
            a = res_opt.output.flatten().astype(np.float32)
            b = res_noopt.output.flatten().astype(np.float32)
            n = min(len(a), len(b))
            diff = np.linalg.norm((a[:n] - b[:n]).astype(np.float64))
            norm = max(np.linalg.norm(b[:n].astype(np.float64)), 1e-8)
            raw  = diff / (norm * self.delta + 1e-8)
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
