"""
TensorFlow Backend — TF-native ONNX interpreter.

  optimized=True  → @tf.function(jit_compile=True)  XLA compilation
  optimized=False → @tf.function                    TF graph mode (no XLA)

The full ONNX graph is expressed in tf ops so XLA actually compiles
and optimizes the computation. Bugs found here are real TF/XLA bugs.
"""
from __future__ import annotations
import logging
import os
from typing import Dict
import numpy as np
import onnx
from onnx import numpy_helper

from .base import BackendBase, BackendResult
from ._onnx_ops import dispatch_op

logger = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class TFBackend(BackendBase):
    name = "tensorflow"

    def is_available(self) -> bool:
        try:
            import tensorflow  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, model: onnx.ModelProto, inputs: Dict[str, np.ndarray],
            optimized: bool = True) -> BackendResult:
        try:
            import tensorflow as tf

            inp_name = model.graph.input[0].name
            out_name = model.graph.output[0].name
            x_np = inputs[inp_name].astype(np.float32)

            # Pre-load initializers as numpy constants (closed-over in tf.function)
            init_np: Dict[str, np.ndarray] = {
                init.name: numpy_helper.to_array(init).copy()
                for init in model.graph.initializer
            }

            # Use tf as the numpy-like module inside dispatch
            # tf ops accept numpy arrays and promote them to tf constants automatically
            import tensorflow.experimental.numpy as tnp
            tnp.experimental_enable_numpy_behavior()

            def model_fn(x_tf):
                values: dict = dict(init_np)
                values[inp_name] = x_tf
                for node in model.graph.node:
                    # Use tf.experimental.numpy which mirrors numpy API exactly
                    results = dispatch_op(node, values, tnp)
                    for name, val in zip(node.output, results):
                        if name:
                            values[name] = val
                return tf.cast(values[out_name], tf.float32)

            if optimized:
                compiled = tf.function(model_fn, jit_compile=True)
            else:
                compiled = tf.function(model_fn)

            x_tf = tf.constant(x_np)
            out = compiled(x_tf).numpy().astype(np.float32)
            return BackendResult(out)

        except NotImplementedError as exc:
            logger.debug("TF unsupported op: %s", exc)
            return BackendResult(None, str(exc), crashed=True)
        except Exception as exc:
            logger.debug("TF (%s) failed: %s", "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
