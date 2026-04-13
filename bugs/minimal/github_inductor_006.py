#!/usr/bin/env python3
"""
Bug ID     : github_inductor_006
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : avgpool ceil mode border
Root cause : Bug: Inductor #100987 — AvgPool ceil_mode border windows divided by full kernel=9, not actual overlap=4.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: Inductor #100987 — AvgPool ceil_mode border windows divided by full kernel=9, not actual overlap=4.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

x = np.ones((1, 1, 6, 6), dtype=np.float32)
# Set bottom-right 2x2 to known values to check border divisor
x[0, 0, 4, 4] = 5.0
x[0, 0, 4, 5] = 6.0
x[0, 0, 5, 4] = 5.0
x[0, 0, 5, 5] = 6.0

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 6, 6])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

node = helper.make_node("AveragePool", ["X"], ["Y"],
    kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1, count_include_pad=0)
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out   = sess.run(None, {"X": x})[0]

# Border window [4:7,4:7] clips to [4:6,4:6]: 4 elements, sum=22, correct avg=5.5
# Inductor bug: divided by 9 (full kernel) → 22/9 ≈ 2.44
correct = 5.5
border  = float(out[0, 0, 2, 2])
print(f"Output shape: {out.shape}  (expected (1,1,3,3) with ceil_mode)")
print(f"Border window [2,2]: got {border:.4f}, expected {correct:.4f}")
print(f"Inductor would give: {22.0/9:.4f}  (wrong: divides by full kernel=9)")
PASS = out.shape == (1, 1, 3, 3) and abs(border - correct) < 0.1
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
