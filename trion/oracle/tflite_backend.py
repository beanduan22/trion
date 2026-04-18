"""
TFLite Backend — converts ONNX to TFLite FlatBuffer via tf.function concrete function.

  optimized=True  → TFLite with DEFAULT optimizations (op fusion, buffer compression)
  optimized=False → TFLite with no optimizations (reference kernels, float32 only)

Surfaces bugs in TFLite's graph simplification and operator conversion passes.
Requires TensorFlow ≥ 2.x.
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


class TFLiteBackend(BackendBase):
    name = "tflite"

    def is_available(self) -> bool:
        try:
            import tensorflow as tf  # noqa: F401
            _ = tf.lite.TFLiteConverter
            return True
        except (ImportError, AttributeError):
            return False

    def run(self, model: onnx.ModelProto, inputs: Dict[str, np.ndarray],
            optimized: bool = True) -> BackendResult:
        try:
            import tensorflow as tf
            import tensorflow.experimental.numpy as tnp
            tnp.experimental_enable_numpy_behavior()

            inp_name = model.graph.input[0].name
            out_name = model.graph.output[0].name
            x_np = inputs[inp_name].astype(np.float32)

            init_np = {
                init.name: numpy_helper.to_array(init).copy()
                for init in model.graph.initializer
            }

            # Build a concrete tf.function from the ONNX graph
            input_spec = tf.TensorSpec(shape=x_np.shape, dtype=tf.float32, name="input")

            @tf.function(input_signature=[input_spec])
            def model_fn(x_tf):
                values: dict = dict(init_np)
                values[inp_name] = x_tf
                for node in model.graph.node:
                    results = dispatch_op(node, values, tnp)
                    for name, val in zip(node.output, results):
                        if name:
                            values[name] = val
                return tf.cast(values[out_name], tf.float32)

            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [model_fn.get_concrete_function()], model_fn
            )
            # Always enforce float32 to avoid quantization-induced divergence.
            # Optimize.DEFAULT without float32 target triggers post-training
            # integer quantization which is expected to diverge — not a compiler
            # bug.  We want graph-level optimizations (op fusion, const folding)
            # while keeping precision comparable to the reference.
            converter.target_spec.supported_types = [tf.float32]
            if optimized:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # unoptimized: no converter.optimizations set (empty = reference ops)
            tflite_flat = converter.convert()

            # Run with TFLite interpreter
            interp = tf.lite.Interpreter(model_content=tflite_flat)
            interp.allocate_tensors()
            inp_det = interp.get_input_details()[0]
            out_det = interp.get_output_details()[0]
            interp.set_tensor(inp_det["index"], x_np)
            interp.invoke()
            out = interp.get_tensor(out_det["index"]).astype(np.float32)
            return BackendResult(out)

        except NotImplementedError as exc:
            logger.debug("TFLite unsupported op: %s", exc)
            return BackendResult(None, str(exc), crashed=True)
        except Exception as exc:
            logger.debug("TFLite (%s) failed: %s", "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
