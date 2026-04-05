"""
PyTorch Eager Reference Backend.
Converts the ONNX model to a PyTorch module via onnx2torch (or falls back to
onnxruntime with graph_optimization_level=DISABLE_ALL as a CPU reference).
"""
from __future__ import annotations
import logging
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)


class PyTorchEagerBackend(BackendBase):
    """
    Reference backend: PyTorch eager execution.

    Execution path (in order of preference):
      1. onnx2torch  → torch model → forward()
      2. onnxruntime (CPU, no optimization) — used as fallback
    """
    name = "pytorch_eager"

    def __init__(self) -> None:
        self._torch_available = self._check_torch()
        self._onnx2torch_available = self._check_onnx2torch()
        self._ort_available = self._check_ort()

    @staticmethod
    def _check_torch() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_onnx2torch() -> bool:
        try:
            import onnx2torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_ort() -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._torch_available or self._ort_available

    def run(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        optimized: bool = True,
    ) -> BackendResult:
        if self._torch_available and self._onnx2torch_available:
            return self._run_onnx2torch(model, inputs)
        if self._ort_available:
            return self._run_ort_no_opt(model, inputs)
        return BackendResult(None, "No suitable backend found", crashed=True)

    def _run_onnx2torch(self, model, inputs) -> BackendResult:
        try:
            import torch
            import onnx2torch
            torch_model = onnx2torch.convert(model)
            torch_model.eval()
            with torch.no_grad():
                input_name = model.graph.input[0].name
                x = torch.from_numpy(inputs[input_name])
                out = torch_model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return BackendResult(out.numpy())
        except Exception as exc:
            logger.debug("onnx2torch failed: %s", exc)
            # fall back
            if self._ort_available:
                return self._run_ort_no_opt(model, inputs)
            return BackendResult(None, str(exc), crashed=True)

    def _run_ort_no_opt(self, model, inputs) -> BackendResult:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            feed = {k: v.astype(np.float32) for k, v in inputs.items()
                    if k in [i.name for i in sess.get_inputs()]}
            output = sess.run(None, feed)[0]
            return BackendResult(output)
        except Exception as exc:
            return BackendResult(None, str(exc), crashed=True)
