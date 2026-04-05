"""
XLA Target Backend.
Converts ONNX → TensorFlow saved model, then runs with jit_compile=True (XLA).
Requires: onnx-tf, tensorflow.
"""
from __future__ import annotations
import logging
import os
import tempfile
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)


class XLABackend(BackendBase):
    name = "xla"

    def is_available(self) -> bool:
        try:
            import tensorflow  # noqa: F401
            import onnx_tf     # noqa: F401
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
            import tensorflow as tf
            from onnx_tf.backend import prepare

            tf_rep = prepare(model)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_path = os.path.join(tmpdir, "model")
                tf_rep.export_graph(export_path)
                loaded = tf.saved_model.load(export_path)
                infer_fn = loaded.signatures["serving_default"]

            input_name = model.graph.input[0].name
            x = tf.constant(inputs[input_name], dtype=tf.float32)

            if optimized:
                @tf.function(jit_compile=True)
                def run_xla(inp):
                    return infer_fn(**{input_name: inp})

                result = run_xla(x)
            else:
                @tf.function(jit_compile=False)
                def run_no_xla(inp):
                    return infer_fn(**{input_name: inp})

                result = run_no_xla(x)

            # Extract the output tensor (first value in the dict)
            if isinstance(result, dict):
                output = list(result.values())[0].numpy()
            else:
                output = result.numpy()

            return BackendResult(output)
        except Exception as exc:
            logger.debug("XLA backend failed: %s", exc)
            return BackendResult(None, str(exc), crashed=True)
