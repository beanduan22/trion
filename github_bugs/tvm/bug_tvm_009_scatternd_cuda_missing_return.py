# Bug: TVM PR #7447 — ScatterND CUDA kernel missing return; thread continued past update, corrupting memory.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

data    = np.zeros((4, 4), dtype=np.float32)
indices = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64)
updates = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [4, 4])
I = helper.make_tensor_value_info("I", TensorProto.INT64, [4, 2])
U = helper.make_tensor_value_info("U", TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])

node  = helper.make_node("ScatterND", inputs=["D", "I", "U"], outputs=["Y"])
graph = helper.make_graph([node], "g", [D, I, U], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"D": data, "I": indices, "U": updates})[0]

expected = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
max_diff = float(np.max(np.abs(out - expected)))

print(f"ScatterND: scatter [1,2,3,4] onto diagonal of 4x4 zeros")
print(f"ORT output:\n{out}")
print(f"Expected:\n{expected}")
print(f"Max diff: {max_diff:.6f}")
print(f"TVM bug: GPU kernel missing 'return' -> thread continued -> memory corruption")
PASS = max_diff < 1e-6
print(f"PASS={PASS}")
