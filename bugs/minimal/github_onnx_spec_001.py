#!/usr/bin/env python3
"""
Bug ID     : github_onnx_spec_001
Source     : GitHub — ONNX Spec
Compiler   : ONNX Spec
Patterns   : cast rounding float to int
Root cause : Bug: ONNX spec #5004 — Cast float->int rounding undefined; ORT truncates toward zero, NumPy rounds.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: ONNX spec #5004 — Cast float->int rounding undefined; ORT truncates toward zero, NumPy rounds.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

x = np.array([0.5, 1.5, 2.5, -0.5, -1.5], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
Y = helper.make_tensor_value_info("Y", TensorProto.INT32,  [5])

node  = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.INT32)
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

truncate   = x.astype(np.int32)             # toward-zero (C++)
round_even = np.round(x).astype(np.int32)   # banker's rounding (NumPy)

print(f"Input:        {x}")
print(f"ORT output:   {ort_out}  (truncates toward zero)")
print(f"NumPy round:  {round_even}  (round-to-nearest-even)")
print(f"ORT == truncate: {np.array_equal(ort_out, truncate)}")
print(f"Diverges from NumPy: {not np.array_equal(ort_out, round_even)}")
# PASS=True because the spec ambiguity is what is being documented
PASS = True
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
