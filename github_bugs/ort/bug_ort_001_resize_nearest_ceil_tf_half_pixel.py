import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: tf_half_pixel_for_nn applied to batch/channel dims in old ORT caused
# batch1 corner to return batch0 values. Fixed in ORT >= 1.8.
x = np.array([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], dtype=np.float32)

scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 2, 2])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1, 4, 4])
node = helper.make_node(
    "Resize", inputs=["X", "", "scales"], outputs=["Y"],
    mode="nearest",
    coordinate_transformation_mode="tf_half_pixel_for_nn",
    nearest_mode="round_prefer_ceil",
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out = sess.run(None, {"X": x})[0]

# Fixed ORT: batch dims must not be transformed; each batch output corner = its own input corner
batch0_ok = out[0, 0, 0, 0] == 1.0
batch1_ok = out[1, 0, 0, 0] == 5.0
print(f"batch0 corner: {out[0,0,0,0]} (expected 1.0)")
print(f"batch1 corner: {out[1,0,0,0]} (expected 5.0)")
PASS = batch0_ok and batch1_ok
print(f"PASS={PASS}")
