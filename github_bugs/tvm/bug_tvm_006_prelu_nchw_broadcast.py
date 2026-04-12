# Bug: TVM PR #7208 — PReLU hardcoded channel axis=1 (NCHW); NHWC slope broadcast applied to H not C.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(23)
x     = np.random.randn(1, 4, 3, 3).astype(np.float32)          # NCHW [1,4,3,3]
slope = np.array([0.1, 0.2, 0.5, 0.9], dtype=np.float32).reshape(4, 1, 1)

X = helper.make_tensor_value_info("X",     TensorProto.FLOAT, [1, 4, 3, 3])
Y = helper.make_tensor_value_info("Y",     TensorProto.FLOAT, [1, 4, 3, 3])
S = numpy_helper.from_array(slope, "slope")

node  = helper.make_node("PRelu", inputs=["X", "slope"], outputs=["Y"])
graph = helper.make_graph([node], "g", [X], [Y], initializer=[S])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

# NumPy reference: PReLU(x) = max(0,x) + slope * min(0,x), broadcast over channels
slope_bc = slope.reshape(1, 4, 1, 1)
ref = np.where(x >= 0, x, slope_bc * x)

max_diff = float(np.max(np.abs(ort_out - ref)))
print(f"Input NCHW shape: {x.shape}, per-channel slopes: {slope.flatten()}")
print(f"ORT output[0,:,0,0]: {ort_out[0,:,0,0]}")
print(f"Ref output[0,:,0,0]: {ref[0,:,0,0]}")
print(f"Max diff ORT vs NumPy: {max_diff:.8f}")
print(f"TVM bug: axis=1 hardcoded — NHWC slope broadcast applies to H instead of C")
PASS = max_diff < 1e-5
print(f"PASS={PASS}")
