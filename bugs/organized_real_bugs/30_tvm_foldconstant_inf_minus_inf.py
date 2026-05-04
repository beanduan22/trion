import sys
import numpy as np
import onnxruntime as ort
import tvm
from onnx import TensorProto, helper
from tvm import relay
from tvm.contrib import graph_executor

x = np.array([1, 2, 3, 4], dtype=np.float32)
g = helper.make_graph(
    [helper.make_node("Mul", ["X", "INF"], ["A"]), helper.make_node("Sub", ["A", "A"], ["Y"])],
    "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
    initializer=[helper.make_tensor("INF", TensorProto.FLOAT, [4], [float("inf")] * 4)],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
ort_out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
mod, params = relay.frontend.from_onnx(m, shape={"X": [4]})
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)
rt = graph_executor.GraphModule(lib["default"](tvm.cpu()))
rt.set_input("X", x)
rt.run()
tvm_out = rt.get_output(0).numpy()
bug = np.isnan(ort_out).all() and not np.isnan(tvm_out).all()
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
