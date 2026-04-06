"""
XLA Backend — JAX-native ONNX interpreter compiled through XLA.

Builds the ONNX computation as a Python function that closes over
pre-loaded numpy initializers, then compiles it with jax.jit so the
full model runs through XLA's optimizer.

  optimized=True  → jax.jit  (XLA: constant folding, op fusion, layout)
  optimized=False → jax.disable_jit (interpreter, no XLA)

Key design: initializers are loaded as numpy arrays BEFORE jax.jit so
they are Python-level closed-over constants. Only the model input is a
traced JAX array. This avoids np.array() on traced values.
"""
from __future__ import annotations
import logging
from typing import Dict
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

from .base import BackendBase, BackendResult
from ._onnx_ops import dispatch_op   # shared op dispatcher

logger = logging.getLogger(__name__)


class XLABackend(BackendBase):
    name = "xla"

    def is_available(self) -> bool:
        try:
            import jax        # noqa: F401
            import jax.numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, model: onnx.ModelProto, inputs: Dict[str, np.ndarray],
            optimized: bool = True) -> BackendResult:
        try:
            import jax
            import jax.numpy as jnp

            inp_name = model.graph.input[0].name
            out_name = model.graph.output[0].name
            x_np = inputs[inp_name].astype(np.float32)

            # Pre-load all initializers as numpy OUTSIDE jax.jit
            init_np: Dict[str, np.ndarray] = {
                init.name: numpy_helper.to_array(init).copy()
                for init in model.graph.initializer
            }

            def model_fn(x_jax):
                # Shallow-copy: numpy values stay numpy (closed-over constants),
                # JAX traced input replaces the model's input slot.
                values: dict = dict(init_np)
                values[inp_name] = x_jax
                for node in model.graph.node:
                    results = dispatch_op(node, values, jnp)
                    for name, val in zip(node.output, results):
                        if name:
                            values[name] = val
                return jnp.asarray(values[out_name], dtype=jnp.float32)

            if optimized:
                jit_fn = jax.jit(model_fn)
                out = np.array(jit_fn(jnp.array(x_np)), dtype=np.float32)
            else:
                with jax.disable_jit():
                    out = np.array(model_fn(jnp.array(x_np)), dtype=np.float32)

            return BackendResult(out)

        except NotImplementedError as exc:
            logger.debug("XLA unsupported op: %s", exc)
            return BackendResult(None, str(exc), crashed=True)
        except Exception as exc:
            logger.debug("XLA (%s) failed: %s", "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
