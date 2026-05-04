import sys
import numpy as np
import onnxruntime as ort
import openvino as ov
from onnx import TensorProto, helper, numpy_helper

rng = np.random.RandomState(42)
x = rng.randn(2, 512).astype(np.float32)
w = rng.randn(512, 64).astype(np.float32)
b = rng.randn(64).astype(np.float32)
r = rng.randn(2, 64).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("MatMul", ["X", "W"], ["A"]), helper.make_node("Add", ["A", "B"], ["C"]), helper.make_node("Relu", ["C"], ["D"]), helper.make_node("Sub", ["D", "R"], ["Y"])],
    "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 512])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 64])],
    initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B"), numpy_helper.from_array(r, "R")],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
mb = m.SerializeToString()
ref = ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
ov_out = ov.Core().compile_model(ov.Core().read_model(mb, b""), "CPU")({"X": x})
bug = float(np.max(np.abs(ref - list(ov_out.values())[0]))) > 0.01
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
