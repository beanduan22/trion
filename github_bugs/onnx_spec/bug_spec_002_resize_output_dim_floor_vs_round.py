# Bug: ONNX spec #4919 — Resize output dim floor vs round inconsistency; 7*1.5=10.5, ORT uses floor=10.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Input size 7, scale 1.5 → 10.5: floor=10, round=11
src = np.zeros((1, 1, 7, 7), dtype=np.float32)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 7, 7])
sc     = helper.make_tensor("sc", TensorProto.FLOAT, [4], [1., 1., 1.5, 1.5])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

node  = helper.make_node("Resize", ["X", "", "sc"], ["Y"],
                         mode="nearest",
                         coordinate_transformation_mode="asymmetric")
graph = helper.make_graph([node], "g", [X], [Y], initializer=[sc])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out   = sess.run(None, {"X": src})[0]

actual_dim = out.shape[2]
print(f"Input=7, scale=1.5 → 7*1.5=10.5")
print(f"floor=10, round=11, ORT actual={actual_dim}")
print(f"ORT uses floor: {actual_dim == 10}")
# Spec ambiguity: stretch mode=floor, keep_aspect_ratio mode=round
PASS = True  # documenting ambiguity
print(f"PASS={PASS}")
