import sys
import numpy as np
import onnxruntime as ort
import openvino as ov
from onnx import TensorProto, helper, numpy_helper

rng = np.random.RandomState(42)
x = rng.uniform(0.001, 0.1, (2, 16)).astype(np.float32)
w = (rng.randn(16, 8) * 10).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("Reciprocal", ["X"], ["A"]), helper.make_node("MatMul", ["A", "W"], ["Y"])],
    "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 16])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8])],
    initializer=[numpy_helper.from_array(w, "W")],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
mb = m.SerializeToString()
ref = ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
ov_out = ov.Core().compile_model(ov.Core().read_model(mb, b""), "CPU")({"X": x})
bug = float(np.max(np.abs(ref - list(ov_out.values())[0]))) > 1.0
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
