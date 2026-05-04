import sys
import numpy as np
import onnxruntime as ort
import openvino as ov
from onnx import TensorProto, helper, numpy_helper

x = np.random.RandomState(0).randn(1, 4, 8, 8).astype(np.float32)
w = np.random.RandomState(1).randn(4, 4, 3, 3).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
    "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
    initializer=[numpy_helper.from_array(w, "w")],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
mb = m.SerializeToString()
ref = ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
ov_out = ov.Core().compile_model(ov.Core().read_model(mb, b""), "CPU")({"x": x})
bug = float(np.max(np.abs(ref - list(ov_out.values())[0]))) > 0.02
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
