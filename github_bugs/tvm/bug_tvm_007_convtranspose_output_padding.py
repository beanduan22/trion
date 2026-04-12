# Bug: TVM PR #7958 — ConvTranspose ignores output_padding; stride=2 + output_padding=[1,1] gave 9x9 not 10x10.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(11)
x = np.random.randn(1, 1, 4, 4).astype(np.float32)
w = np.random.randn(1, 1, 3, 3).astype(np.float32)

X     = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y     = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
W_init = numpy_helper.from_array(w, "W")

# output_size = (input-1)*stride - 2*pad + kernel + output_padding = 3*2+3+1 = 10
node = helper.make_node(
    "ConvTranspose", inputs=["X", "W"], outputs=["Y"],
    strides=[2, 2],
    output_padding=[1, 1],
    pads=[0, 0, 0, 0],
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[W_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"X": x})[0]

expected_shape  = (1, 1, 10, 10)
wrong_shape     = (1, 1, 9, 9)
shape_ok = out.shape == expected_shape

print(f"Input: {x.shape}, stride=2, output_padding=[1,1]")
print(f"ORT output shape:             {out.shape}")
print(f"Expected with output_padding: {expected_shape}")
print(f"TVM bug: always produced:     {wrong_shape}")
print(f"Shape correct: {shape_ok}")
PASS = shape_ok
print(f"PASS={PASS}")
