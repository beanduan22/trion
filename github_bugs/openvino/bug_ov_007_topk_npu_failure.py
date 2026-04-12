# OpenVINO Bug #29297 - TopK NPU throws ZE_RESULT_ERROR_UNKNOWN for large K
# https://github.com/openvinotoolkit/openvino/issues/29297
# OV bug: NPU plugin failed for TopK with K=128 from N=5040 candidates
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(0)
N   = 5040
K   = 128
x   = np.random.randn(1, N).astype(np.float32)
k_v = np.array([K], dtype=np.int64)

X   = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, N])
K_  = helper.make_tensor_value_info("K", TensorProto.INT64, [1])
V   = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, K])
I   = helper.make_tensor_value_info("I", TensorProto.INT64,  [1, K])
node  = helper.make_node("TopK", ["X", "K"], ["V", "I"], axis=1, largest=1, sorted=1)
graph = helper.make_graph([node], "g", [X, K_], [V, I])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
vals, idxs = sess.run(None, {"X": x, "K": k_v})

sorted_ok  = bool(np.all(vals[0, :-1] >= vals[0, 1:]))
indices_ok = bool(np.allclose(x[0, idxs[0]], vals[0]))
print(f"TopK K={K} from N={N}: output shape {vals.shape}")
print(f"top-3 values: {vals[0, :3]}")
print(f"sorted desc:  {sorted_ok}")
print(f"indices match: {indices_ok}")
print(f"PASS={sorted_ok and indices_ok}")
