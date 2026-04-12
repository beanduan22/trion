import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: ORT JSEP ScatterND race with duplicate indices (PR#23755). CPU is deterministic (last-write wins).
data    = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
indices = np.array([[2], [2], [4]], dtype=np.int64)  # duplicate index 2
updates = np.array([10.0, 20.0, 30.0], dtype=np.float32)

D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [5])
I = helper.make_tensor_value_info("I", TensorProto.INT64, [3, 1])
U = helper.make_tensor_value_info("U", TensorProto.FLOAT, [3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5])

node = helper.make_node("ScatterND", inputs=["D","I","U"], outputs=["Y"],
                        reduction="none")
graph = helper.make_graph([node], "g", [D, I, U], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out = sess.run(None, {"D": data, "I": indices, "U": updates})[0]

# CPU: sequential scatter — last update wins → index 2 gets 20.0 (second write)
expected = np.array([1.0, 2.0, 20.0, 4.0, 30.0], dtype=np.float32)
max_diff = float(np.max(np.abs(out - expected)))
print(f"Data: {data}")
print(f"Indices (with duplicate): {indices.flatten()}")
print(f"Updates: {updates}")
print(f"Expected (last-write-wins): {expected}")
print(f"ORT CPU output:             {out}")
print(f"Max diff: {max_diff:.6f}")
print(f"PASS={max_diff < 1e-5}  (CPU deterministic; JSEP race was on WebGPU)")
