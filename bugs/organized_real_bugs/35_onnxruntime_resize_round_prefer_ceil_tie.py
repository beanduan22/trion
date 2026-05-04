import sys
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

x = np.linspace(0, 1, 20, dtype=np.float32).reshape(1, 1, 1, 20)
g = helper.make_graph(
    [helper.make_node("Resize", ["X", "", "", "sizes"], ["Y"], mode="nearest", coordinate_transformation_mode="half_pixel", nearest_mode="round_prefer_ceil")],
    "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 20])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 6])],
    initializer=[helper.make_tensor("sizes", TensorProto.INT64, [4], [1, 1, 1, 6])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0].reshape(-1)
bug = abs(float(out[4]) - float(x.reshape(-1)[15])) > 1e-4
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
