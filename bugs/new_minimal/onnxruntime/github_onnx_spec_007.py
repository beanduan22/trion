#!/usr/bin/env python3
"""
Bug ID     : github_onnx_spec_007
Source     : GitHub — ONNX Spec
Compiler   : ONNX Spec
Patterns   : resize nearest round prefer ceil edge
Root cause : Bug: Resize nearest half_pixel round_prefer_ceil 20->6: ORT returned wrong index at element 4 (spec #4583).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: Resize nearest half_pixel round_prefer_ceil 20->6: ORT returned wrong index at element 4 (spec #4583).
# half_pixel: x_orig = (i+0.5)*(20/6)-0.5 → indices [1,4,8,11,15,18]
# Expected element 4: index 15, value = 15/19 ≈ 0.789
x = np.linspace(0, 1, 20, dtype=np.float32).reshape(1, 1, 1, 20)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 20])
sizes  = helper.make_tensor("sizes", TensorProto.INT64, [4], [1, 1, 1, 6])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 6])
node   = helper.make_node("Resize", ["X","","","sizes"], ["Y"],
                          mode="nearest",
                          coordinate_transformation_mode="half_pixel",
                          nearest_mode="round_prefer_ceil")
graph  = helper.make_graph([node], "g", [X], [Y], initializer=[sizes])
model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0].flatten()

# Correct indices for half_pixel + round_prefer_ceil: [1, 4, 8, 11, 15, 18]
expected = x.flatten()[[1, 4, 8, 11, 15, 18]]

print(f"ORT  output: {ort_out}")
print(f"Expected:    {expected}")
print(f"Bug (elem 4): ORT={ort_out[4]:.4f}, expected={expected[4]:.4f}")
PASS = np.allclose(ort_out, expected, atol=1e-5)
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
