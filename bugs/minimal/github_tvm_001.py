#!/usr/bin/env python3
"""
Bug ID     : github_tvm_001
Source     : GitHub — TVM Relay
Compiler   : TVM Relay
Patterns   : batchnorm simplifyinference skip
Root cause : Bug: TVM #6852 — SimplifyInference skips BN when output stored before indexing; train-mode stats used at inference.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: TVM #6852 — SimplifyInference skips BN when output stored before indexing; train-mode stats used at inference.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(42)
x = np.random.randn(2, 4, 4, 4).astype(np.float32)
scale = np.ones(4, dtype=np.float32)
bias  = np.zeros(4, dtype=np.float32)
mean  = np.full(4, 0.5, dtype=np.float32)   # non-trivial running mean
var   = np.full(4, 2.0, dtype=np.float32)   # non-trivial running var

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 4, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 4, 4])
S = numpy_helper.from_array(scale, "scale")
B = numpy_helper.from_array(bias,  "bias")
M = numpy_helper.from_array(mean,  "mean")
V = numpy_helper.from_array(var,   "var")

node = helper.make_node(
    "BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"],
    epsilon=1e-5, momentum=0.9,
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[S, B, M, V])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
model.graph.node[0].attribute  # eval mode: uses running mean/var

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

# NumPy reference for eval-mode BN (uses running mean/var, not batch stats)
ref = (x - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + 1e-5)
ref = ref * scale[None, :, None, None] + bias[None, :, None, None]

max_diff = float(np.max(np.abs(ort_out - ref)))
print(f"ORT BN eval output[0,0,0,:3]: {ort_out[0,0,0,:3]}")
print(f"NumPy ref  output[0,0,0,:3]:  {ref[0,0,0,:3]}")
print(f"Max diff (eval mode correct): {max_diff:.6f}")
print(f"TVM bug: SimplifyInference skips BN -> train-mode stats used -> wrong outputs")
PASS = max_diff < 1e-4
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
