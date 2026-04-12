# OpenVINO Bug #5505 - Wrong output in Concat->Resize->Concat (in-place buffer reuse)
# https://github.com/openvinotoolkit/openvino/issues/5505
# OV bug: memory manager reused Concat output buffer as Resize input, corrupting data
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(3)
a = np.random.rand(1, 2, 2, 2).astype(np.float32)
b = np.random.rand(1, 2, 2, 2).astype(np.float32)
c = np.random.rand(1, 4, 4, 4).astype(np.float32)

A      = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2, 2, 2])
B      = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 2, 2, 2])
C      = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4, 4, 4])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 4, 4])
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), "scales")

cat1   = helper.make_node("Concat", ["A", "B"], ["cat1"], axis=1)
resize = helper.make_node("Resize", ["cat1", "", "scales"], ["res"],
                          mode="nearest",
                          coordinate_transformation_mode="asymmetric")
cat2   = helper.make_node("Concat", ["res", "C"], ["Y"], axis=1)
graph  = helper.make_graph([cat1, resize, cat2], "g", [A, B, C], [Y], initializer=[scales])
model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"A": a, "B": b, "C": c})[0]

# Expected first 4 channels: nearest upsample of concat(a,b)
expected_first4 = np.repeat(np.repeat(np.concatenate([a, b], axis=1), 2, axis=2), 2, axis=3)
max_diff = float(np.max(np.abs(out[:, :4] - expected_first4)))
print(f"output shape: {out.shape}")
print(f"out[0,0,0,:2]:      {out[0,0,0,:2]}")
print(f"expected[0,0,0,:2]: {expected_first4[0,0,0,:2]}")
print(f"max_diff first 4ch: {max_diff:.2e}")
print(f"PASS={max_diff < 1e-5}")
