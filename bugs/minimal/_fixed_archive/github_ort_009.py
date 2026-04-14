#!/usr/bin/env python3
"""
Bug ID     : github_ort_009
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : optimizer shape node sharing
Root cause : Two tensors with the same shape but different values; Shape nodes must NOT be shared
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
"""
ORT Bug #20951 — ORT_ENABLE_EXTENDED optimizer incorrectly shares Shape nodes
https://github.com/microsoft/onnxruntime/issues/20951
Status: Closed

Root cause: The optimizer shared Shape output nodes for inputs with identical
symbolic dimensions but different runtime values, feeding the wrong tensor
value to downstream operations.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Two tensors with the same shape but different values; Shape nodes must NOT be shared
np.random.seed(5)
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 4).astype(np.float32) * 10.0  # clearly different scale

A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])

# Graph: Gather shape info then use in computation
# Shape(A) -> Reshape(B) then Add(A, reshaped_B)
shape_a = helper.make_node("Shape", ["A"], ["shape_a"])
shape_b = helper.make_node("Shape", ["B"], ["shape_b"])
# Flatten both, then concatenate shape tensors as a sanity check
flat_a = helper.make_node("Flatten", ["A"], ["flat_a"], axis=0)
flat_b = helper.make_node("Flatten", ["B"], ["flat_b"], axis=0)
add    = helper.make_node("Add",  ["flat_a", "flat_b"], ["Y"])

graph  = helper.make_graph([shape_a, shape_b, flat_a, flat_b, add], "shape_share", [A, B], [Y])
model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(),
                            providers=["CPUExecutionProvider"])
out = sess.run(None, {"A": a, "B": b})[0]
expected = (a + b).flatten()
max_err = float(np.max(np.abs(out - expected)))
print(f"A[0,:3]: {a[0,:3]}")
print(f"B[0,:3]: {b[0,:3]}")
print(f"Expected A+B flat [0:3]: {expected[:3]}")
print(f"ORT output [0:3]:        {out[:3]}")
print(f"Max abs error: {max_err:.6f}")
print(f"PASS={max_err < 1e-5}")

PASS = max_err < 1e-5
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
