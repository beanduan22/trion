"""
OpenVINO Backend — Intel's inference engine.

Converts ONNX directly to OpenVINO IR and runs with the CPU plugin.

  optimized=True  → Core().compile_model() with CPU plugin (full optimization)
  optimized=False → compile_model with hints={"PERFORMANCE_HINT": "LATENCY"} disabled

OpenVINO applies constant folding, quantization-aware inference, and
CPU-specific operator fusion. Bugs here reflect real OpenVINO compiler issues.
Requires: pip install openvino
"""
from __future__ import annotations
import logging
import tempfile
import os
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)


class OpenVINOBackend(BackendBase):
    name = "openvino"

    def is_available(self) -> bool:
        try:
            import openvino  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, model: onnx.ModelProto, inputs: Dict[str, np.ndarray],
            optimized: bool = True) -> BackendResult:
        try:
            from openvino import Core, convert_model
            import openvino as ov

            inp_name = model.graph.input[0].name
            x_np = inputs[inp_name].astype(np.float32)

            # Write ONNX to a temp file (convert_model accepts a path or bytes)
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                f.write(model.SerializeToString())
                onnx_path = f.name

            try:
                ov_model = convert_model(onnx_path)
            finally:
                os.unlink(onnx_path)

            core = Core()
            if optimized:
                config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            else:
                # Disable most optimizations — use "accuracy" mode if available
                config = {"PERFORMANCE_HINT": "LATENCY",
                          "INFERENCE_PRECISION_HINT": "f32"}

            compiled = core.compile_model(ov_model, "CPU", config)
            infer_req = compiled.create_infer_request()

            # Get input/output info
            ov_input  = compiled.input(0)
            ov_output = compiled.output(0)

            infer_req.infer({ov_input: x_np})
            out = infer_req.get_output_tensor(ov_output.index).data.astype(np.float32)
            return BackendResult(out)

        except Exception as exc:
            logger.debug("OpenVINO (%s) failed: %s", "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
