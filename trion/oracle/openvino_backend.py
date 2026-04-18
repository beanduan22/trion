"""
OpenVINO Backend — Intel's inference engine.

Converts ONNX directly to OpenVINO IR and runs with the CPU plugin.

  optimized=True  → compile_model with default config (constant folding,
                     op fusion, layout transforms — full OV optimizer)
  optimized=False → compile_model with INFERENCE_PRECISION_HINT=f32 and
                     CACHE_DIR disabled; passes a pre-serialized model
                     through ov.serialize so no shape/type inference is
                     re-run (disables OV graph rewriter passes).

Bugs here reflect real OpenVINO IR compiler issues.
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
                # Full OV optimization pipeline (default): constant folding,
                # op fusion, CPU-specific layout transforms.
                config = {"INFERENCE_PRECISION_HINT": "f32"}
            else:
                # Minimal compilation: force f32, disable threading variability,
                # and set performance hint to LATENCY (single-thread) to suppress
                # any parallelism-driven nondeterminism.  OV has no public
                # "disable all passes" API, so we rely on f32 + single-thread
                # being a stable reference and measure divergence via s_diff
                # (target_opt vs pytorch_eager) rather than opt-vs-noopt.
                config = {
                    "INFERENCE_PRECISION_HINT": "f32",
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": "1",
                }

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
