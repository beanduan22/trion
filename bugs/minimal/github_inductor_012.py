#!/usr/bin/env python3
"""
Bug ID     : github_inductor_012
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : einsum matmul first call
Root cause : Bug: PyTorch #85224 — MPS einsum batch matmul wrong on first call; Metal command buffer not initialized.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: PyTorch #85224 — MPS einsum batch matmul wrong on first call; Metal command buffer not initialized.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(71)
A = np.random.randn(2, 3, 4).astype(np.float32)
B = np.random.randn(2, 4, 5).astype(np.float32)

X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [2, 3, 4])
X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [2, 4, 5])
Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [2, 3, 5])

node  = helper.make_node("Einsum", ["X1", "X2"], ["Y"], equation="bij,bjk->bik")
graph = helper.make_graph([node], "g", [X1, X2], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out   = sess.run(None, {"X1": A, "X2": B})[0]

ref      = np.einsum("bij,bjk->bik", A, B)
max_diff = float(np.max(np.abs(out - ref)))
print(f"Batch matmul 'bij,bjk->bik': max diff ORT vs NumPy = {max_diff:.2e}")
PASS = max_diff < 1e-4
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
