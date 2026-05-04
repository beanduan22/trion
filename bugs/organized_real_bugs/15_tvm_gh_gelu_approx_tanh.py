import sys
import numpy as np
import onnxruntime as ort
import tvm
from onnx import TensorProto, helper
from tvm import relay
from tvm.contrib import graph_executor

x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
g = helper.make_graph(
    [helper.make_node("Gelu", ["X"], ["Y"], approximate="tanh")],
    "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [7])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [7])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
ort_out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
mod, params = relay.frontend.from_onnx(m, shape={"X": [7]})
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)
rt = graph_executor.GraphModule(lib["default"](tvm.cpu()))
rt.set_input("X", x)
rt.run()
tvm_out = rt.get_output(0).numpy()
bug = float(np.max(np.abs(ort_out - tvm_out))) > 1e-4
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
