# Bug: ONNX spec #2611 — CumSum axis marked optional in schema; absent axis causes undefined behavior.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

x = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=np.float32)
ax_arr = np.array(1, dtype=np.int64)

X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [2, 4])
Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [2, 4])
init = [numpy_helper.from_array(ax_arr, "Ax")]
node = helper.make_node("CumSum", ["X", "Ax"], ["Y"], exclusive=0, reverse=0)
graph = helper.make_graph([node], "cs", [X], [Y], initializer=init)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out = sess.run(None, {"X": x})[0]

np_ref = np.cumsum(x, axis=1)
max_err = float(np.max(np.abs(out - np_ref)))
print(f"Input:\n{x}")
print(f"CumSum axis=1 output:\n{out}")
print(f"NumPy ref:\n{np_ref}")
print(f"Max err: {max_err:.2e}")
PASS = max_err < 1e-4
print(f"PASS={PASS}")
