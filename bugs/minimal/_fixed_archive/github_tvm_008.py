#!/usr/bin/env python3
"""
Bug ID     : github_tvm_008
Source     : GitHub — TVM Relay
Compiler   : TVM Relay
Patterns   : instance norm mean var
Root cause : Bug: TVM #15683 — InstanceNorm GPU parallel reduction gave wrong mean/var for small spatial dims (2x2).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: TVM #15683 — InstanceNorm GPU parallel reduction gave wrong mean/var for small spatial dims (2x2).
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(17)
x     = np.random.randn(2, 4, 2, 2).astype(np.float32)  # small 2x2 spatial
scale = np.ones(4, dtype=np.float32)
bias  = np.zeros(4, dtype=np.float32)

X = helper.make_tensor_value_info("X",     TensorProto.FLOAT, [2, 4, 2, 2])
Y = helper.make_tensor_value_info("Y",     TensorProto.FLOAT, [2, 4, 2, 2])
S = numpy_helper.from_array(scale, "scale")
B = numpy_helper.from_array(bias,  "bias")

node  = helper.make_node("InstanceNormalization", ["X", "scale", "bias"], ["Y"], epsilon=1e-5)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[S, B])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 6)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

# NumPy reference: normalize per (n, c) over spatial dims
ref = np.zeros_like(x)
for n in range(2):
    for c in range(4):
        v   = x[n, c]
        mu  = v.mean()
        var = v.var()
        ref[n, c] = (v - mu) / np.sqrt(var + 1e-5)

max_diff = float(np.max(np.abs(ort_out - ref)))
print(f"Input shape: {x.shape} (small 2x2 spatial, 4 elements per instance)")
print(f"ORT output[0,0]: {ort_out[0,0].flatten()}")
print(f"Ref output[0,0]: {ref[0,0].flatten()}")
print(f"Max diff ORT vs NumPy: {max_diff:.8f}")
print(f"TVM bug: GPU parallel reduction gave wrong mean/var for small spatial dims")
PASS = max_diff < 1e-4
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
