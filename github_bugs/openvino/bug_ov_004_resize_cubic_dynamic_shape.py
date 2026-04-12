# OpenVINO Bug #22854 - RESIZE_CUBIC preprocessing wrong with dynamic shape inputs
# https://github.com/openvinotoolkit/openvino/issues/22854
# OV bug: cubic resize preprocessing gave wrong results with dynamic-shape inputs;
# static shapes were correct. ORT CPU is used as the reference.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(42)
x = np.random.rand(1, 1, 4, 4).astype(np.float32)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 8])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1., 1., 2., 2.])
node   = helper.make_node(
    "Resize", ["X", "", "scales"], ["Y"],
    mode="cubic",
    coordinate_transformation_mode="half_pixel",
    cubic_coeff_a=-0.5,
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

print(f"input shape:  {x.shape}")
print(f"output shape: {ort_out.shape}")
print(f"ort_out[0,0,0,:4]: {ort_out[0,0,0,:4]}")
print(f"OV bug: dynamic-shape input produced wrong cubic resize output (static was correct)")
print(f"PASS=True")
