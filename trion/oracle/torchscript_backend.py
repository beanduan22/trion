"""
TorchScript Target Backend.

Converts ONNX → PyTorch (via onnx2torch) then:
  optimized=True  → torch.jit.trace + torch.jit.optimize_for_inference
  optimized=False → torch.jit.trace (no inference optimization)

This exercises the TorchScript compiler pipeline:
  - Dead-code elimination
  - Constant propagation
  - Op fusion via TorchScript passes
  - inline / peephole optimizations
"""
from __future__ import annotations
import logging
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)


class TorchScriptBackend(BackendBase):
    name = "torchscript"

    def is_available(self) -> bool:
        try:
            import torch        # noqa: F401
            import onnx2torch   # noqa: F401
            return True
        except ImportError:
            return False

    def run(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        optimized: bool = True,
    ) -> BackendResult:
        try:
            import torch
            import onnx2torch

            torch_model = onnx2torch.convert(model)
            torch_model.eval()

            input_name = model.graph.input[0].name
            x = torch.from_numpy(inputs[input_name].astype(np.float32))

            # Trace with a concrete example input
            with torch.no_grad():
                scripted = torch.jit.trace(torch_model, (x,), strict=False)

            if optimized:
                # Apply inference optimizations: op folding, layout opts, etc.
                scripted = torch.jit.optimize_for_inference(scripted)

            with torch.no_grad():
                out = scripted(x)

            if isinstance(out, (list, tuple)):
                out = out[0]
            return BackendResult(out.detach().cpu().numpy())

        except Exception as exc:
            logger.debug("TorchScript (%s) failed: %s",
                         "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
