"""
TVM Target Backend.
Imports ONNX via tvm.relay.frontend.from_onnx, compiles with opt_level 0 or 3.
"""
from __future__ import annotations
import io
import logging
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)


class TVMBackend(BackendBase):
    name = "tvm"

    def __init__(self, target: str = "llvm", opt_level: int = 3) -> None:
        self.target    = target
        self.opt_level = opt_level

    def is_available(self) -> bool:
        try:
            import tvm  # noqa: F401
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
            import tvm
            from tvm import relay
            import tvm.relay.testing
            from tvm.contrib import graph_executor

            input_name  = model.graph.input[0].name
            input_array = inputs[input_name]
            shape_dict  = {input_name: input_array.shape}

            # Import ONNX → Relay IR
            mod, params = relay.frontend.from_onnx(model, shape_dict)

            level = self.opt_level if optimized else 0
            with tvm.transform.PassContext(opt_level=level):
                lib = relay.build(mod, target=self.target, params=params)

            dev = tvm.device(self.target, 0)
            m   = graph_executor.GraphModule(lib["default"](dev))
            m.set_input(input_name, tvm.nd.array(input_array.astype("float32"), dev))
            m.run()
            output = m.get_output(0).numpy()
            return BackendResult(output)
        except Exception as exc:
            logger.debug("TVM (%s) failed: %s", "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
