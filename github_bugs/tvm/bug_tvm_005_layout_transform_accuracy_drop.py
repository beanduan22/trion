"""
TVM Bug #7563 — Accuracy drop with batch>1 at opt_level=3 (NCHWc layout transform)
https://github.com/apache/tvm/issues/7563
Status: Closed

Root cause: AlterOpLayout (NCHWc) + CanonicalizeOps at opt_level=3 produced
incorrect per-sample outputs for batch>1 on LLVM targets. Running the same sample
in batch=1 vs batch=2 gave different results — only batch=1 was correct.
This shows PyTorch reference: same sample must give identical output regardless of batch size.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(42)
# Dilated conv + regular conv (branch merge) — triggers the NCHWc bug
x = np.random.randn(2, 4, 8, 8).astype(np.float32)
w1 = np.random.randn(8, 4, 3, 3).astype(np.float32)  # regular conv
wd = np.random.randn(8, 4, 3, 3).astype(np.float32)  # dilated conv
w2 = np.random.randn(4, 16, 1, 1).astype(np.float32) # 1x1 merge

X   = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [2, 4, 8, 8])
Y   = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)
W1  = numpy_helper.from_array(w1, "W1")
WD  = numpy_helper.from_array(wd, "WD")
W2  = numpy_helper.from_array(w2, "W2")

conv1 = helper.make_node("Conv", ["X","W1"], ["c1"], pads=[1,1,1,1])
conv_d= helper.make_node("Conv", ["X","WD"], ["cd"], pads=[2,2,2,2], dilations=[2,2])
cat   = helper.make_node("Concat", ["c1","cd"], ["cat"], axis=1)
conv2 = helper.make_node("Conv", ["cat","W2"], ["Y"])

graph = helper.make_graph([conv1, conv_d, cat, conv2], "g", [X], [Y],
                          initializer=[W1, WD, W2])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

sess   = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out_b2 = sess.run(None, {"X": x})[0]           # batch=2
out_b1 = sess.run(None, {"X": x[:1]})[0]       # same first sample, batch=1

max_diff = float(np.max(np.abs(out_b2[:1] - out_b1)))
print(f"ORT batch=1 output[0,0,0,:3]: {out_b1[0,0,0,:3]}")
print(f"ORT batch=2 output[0,0,0,:3]: {out_b2[0,0,0,:3]}")
print(f"Max diff (same sample, batch=1 vs batch=2): {max_diff:.6f}")
print(f"TVM bug: NCHWc layout at opt_level=3 gave wrong batch=2 results vs batch=1")
PASS = max_diff < 1e-4
print(f"PASS={PASS}")
