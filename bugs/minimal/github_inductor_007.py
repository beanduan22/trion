#!/usr/bin/env python3
"""
Bug ID     : github_inductor_007
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : avgpool single element gpu
Root cause : Bug: Inductor #143720 — AvgPool GPU kernel returned undivided sum instead of sum/kernel_area.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: Inductor #143720 — AvgPool GPU kernel returned undivided sum instead of sum/kernel_area.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(43)
x = np.random.randn(1, 1, 4, 4).astype(np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

node = helper.make_node("AveragePool", ["X"], ["Y"],
    kernel_shape=[2, 2], strides=[2, 2], count_include_pad=1)
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out   = sess.run(None, {"X": x})[0]

# Reference: non-overlapping 2x2 windows, each averaged over 4 elements
ref = np.array([[[[x[0,0,r*2:r*2+2, c*2:c*2+2].mean()
                   for c in range(2)] for r in range(2)]]])
max_diff = float(np.max(np.abs(out - ref)))
print(f"ORT output[0,0]:  {out[0,0]}")
print(f"NumPy ref[0,0]:   {ref[0,0]}")
print(f"Max diff: {max_diff:.2e}")
PASS = max_diff < 1e-5
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
