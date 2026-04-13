#!/usr/bin/env python3
"""
Bug ID     : github_ort_017
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : einsum uppercase rejection
Root cause : Bug: ORT Einsum rejected uppercase labels like "BIJ,BJK->BIK" (issue #4944).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: ORT Einsum rejected uppercase labels like "BIJ,BJK->BIK" (issue #4944).
np.random.seed(5)
A = np.random.randn(2, 3, 4).astype(np.float32)
B = np.random.randn(2, 4, 5).astype(np.float32)

X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [2, 3, 4])
X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [2, 4, 5])
Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)

# lowercase (always worked)
n_lower = helper.make_node("Einsum", ["X1","X2"], ["Y"], equation="bij,bjk->bik")
g_lower = helper.make_graph([n_lower], "g", [X1, X2], [Y])
m_lower = helper.make_model(g_lower, opset_imports=[helper.make_opsetid("", 12)])
sess_lower = ort.InferenceSession(m_lower.SerializeToString(), providers=["CPUExecutionProvider"])
out_lower  = sess_lower.run(None, {"X1": A, "X2": B})[0]

# uppercase (was rejected in old ORT)
n_upper = helper.make_node("Einsum", ["X1","X2"], ["Y"], equation="BIJ,BJK->BIK")
g_upper = helper.make_graph([n_upper], "g", [X1, X2], [Y])
m_upper = helper.make_model(g_upper, opset_imports=[helper.make_opsetid("", 12)])
sess_upper = ort.InferenceSession(m_upper.SerializeToString(), providers=["CPUExecutionProvider"])
out_upper  = sess_upper.run(None, {"X1": A, "X2": B})[0]

max_diff = float(np.max(np.abs(out_lower - out_upper)))
print(f"lowercase output[0,0,:3]: {out_lower[0,0,:3]}")
print(f"uppercase output[0,0,:3]: {out_upper[0,0,:3]}")
print(f"Max diff: {max_diff:.8f}")
PASS = max_diff < 1e-5
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
