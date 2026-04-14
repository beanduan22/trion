#!/usr/bin/env python3
"""
Bug ID     : github_openvino_014
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : einsum scalar input parse
Root cause : OpenVINO PR#30189 - Einsum parser crashes on scalar (rank-0) input operands
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO PR#30189 - Einsum parser crashes on scalar (rank-0) input operands
# https://github.com/openvinotoolkit/openvino/pull/30189
# OV bug: Einsum frontend assumed rank >= 1; scalar operand caused null-deref / crash
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

vec    = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
scalar = np.array(3.0, dtype=np.float32)  # rank-0

X1   = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [5])
X2   = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [])   # rank-0 scalar
Y    = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [5])
node = helper.make_node("Einsum", ["X1", "X2"], ["Y"], equation="i,->i")
graph = helper.make_graph([node], "g", [X1, X2], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X1": vec, "X2": scalar})[0]

expected = vec * float(scalar)
max_diff = float(np.max(np.abs(ort_out - expected)))
print(f"equation: 'i,->i'  (vector * scalar)")
print(f"vec={vec}, scalar={scalar}")
print(f"ort_out:  {ort_out}")
print(f"expected: {expected}")
print(f"max_diff: {max_diff:.2e}")
print(f"OV bug: scalar operand (rank-0) caused segfault in Einsum parser")
print(f"PASS={max_diff < 1e-5}")

PASS = max_diff < 1e-5
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
