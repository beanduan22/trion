"""
TVM Backend — JAX-JIT ONNX interpreter (Apache TVM substitute).

Apache TVM has no Python 3.13 wheels. This backend uses the shared JAX
ONNX interpreter with jax.jit, which exercises the same XLA optimization
passes (constant folding, op fusion, layout transforms) that TVM's relay
optimizer targets.

  optimized=True  → jax.jit  (XLA-compiled, like TVM opt_level=3)
  optimized=False → jax.disable_jit (interpreter, like TVM opt_level=0)

When Apache TVM wheels become available for Python 3.13, replace this
with the real TVM relay frontend.
"""
from __future__ import annotations
import logging
from typing import Dict
import numpy as np
import onnx
from onnx import numpy_helper

from .base import BackendBase, BackendResult
from ._onnx_ops import dispatch_op

logger = logging.getLogger(__name__)


class TVMBackend(BackendBase):
    name = "tvm"

    def __init__(self, target: str = "llvm", opt_level: int = 3) -> None:
        self.target    = target
        self.opt_level = opt_level

    def is_available(self) -> bool:
        try:
            import jax       # noqa: F401
            import jax.numpy # noqa: F401
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

            init_np: Dict[str, np.ndarray] = {
                init.name: numpy_helper.to_array(init).copy()
                for init in model.graph.initializer
            }

            def model_fn(x_jax):
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
            logger.debug("TVM unsupported op: %s", exc)
            return BackendResult(None, str(exc), crashed=True)
        except Exception as exc:
            logger.debug("TVM (%s) failed: %s", "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
