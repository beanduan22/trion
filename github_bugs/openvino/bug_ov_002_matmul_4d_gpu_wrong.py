# OpenVINO Bug #28881 - 4D MatMul FP32 GPU plugin wrong output
# https://github.com/openvinotoolkit/openvino/issues/28881
# OV bug: [batch,2,1,16] x [batch,2,16,1] gave wrong results on GPU for certain batch sizes
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(7)
A = np.random.randn(4, 2, 1, 16).astype(np.float32)
B = np.random.randn(4, 2, 16, 1).astype(np.float32)

AI = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 2, 1, 16])
BI = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 2, 16, 1])
Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 2, 1, 1])
node = helper.make_node("MatMul", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [AI, BI], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"A": A, "B": B})[0]

expected = np.matmul(A, B)

max_diff = float(np.max(np.abs(ort_out - expected)))
print(f"A shape: {A.shape}, B shape: {B.shape}, out shape: {ort_out.shape}")
print(f"ort_out[0,0,0,0]: {ort_out[0,0,0,0]:.5f}")
print(f"expected[0,0,0,0]: {expected[0,0,0,0]:.5f}")
print(f"max_diff: {max_diff:.2e}")
print(f"PASS={max_diff < 1e-4}")
