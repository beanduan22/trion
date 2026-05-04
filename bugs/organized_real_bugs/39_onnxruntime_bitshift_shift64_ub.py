import numpy as np
import sys
import onnxruntime as ort
from onnx import TensorProto, helper

g = helper.make_graph(
    [helper.make_node("BitShift", ["x", "n"], ["y"], direction="RIGHT")],
    "g",
    [
        helper.make_tensor_value_info("x", TensorProto.UINT64, [1, 2]),
        helper.make_tensor_value_info("n", TensorProto.UINT64, [1, 2]),
    ],
    [helper.make_tensor_value_info("y", TensorProto.UINT64, [1, 2])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11)])
m.ir_version = 8
out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(
    None, {"x": np.array([[1000, 255]], dtype=np.uint64), "n": np.array([[64, 64]], dtype=np.uint64)}
)[0]
bug = out.tolist() != [[0, 0]]
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
