"""
ONNX Runtime Target Backend.
Supports optimized and unoptimized execution to measure Δ_opt.
"""
from __future__ import annotations
import logging
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)


class ONNXRuntimeBackend(BackendBase):
    name = "onnxruntime"

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
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
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if optimized:
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            else:
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                sess = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=opts,
                    providers=providers,
                )
            except Exception:
                sess = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )

            input_names = {i.name for i in sess.get_inputs()}
            feed = {k: v.astype(np.float32)
                    for k, v in inputs.items() if k in input_names}
            output = sess.run(None, feed)[0]
            return BackendResult(output)
        except Exception as exc:
            logger.debug("OnnxRuntime (%s) failed: %s",
                         "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
