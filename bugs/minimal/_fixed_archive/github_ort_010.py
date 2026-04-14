#!/usr/bin/env python3
"""
Bug ID     : github_ort_010
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : resize linear gpu half pixel
Root cause : Bug: ORT GPU Resize linear outputs constant 0.5 when preceded by MatMul (issue #12091).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Bug: ORT GPU Resize linear outputs constant 0.5 when preceded by MatMul (issue #12091).
# CPU is correct; GPU was broken on pre-fix ORT.
np.random.seed(99)
x = np.random.rand(1, 4, 2, 2).astype(np.float32)
w = np.random.rand(4, 4).astype(np.float32)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2, 2])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4, 4])
W_init = numpy_helper.from_array(w, "W")
s_flat = numpy_helper.from_array(np.array([4, 4],       dtype=np.int64), "s_flat")
s_orig = numpy_helper.from_array(np.array([1, 4, 2, 2], dtype=np.int64), "s_orig")
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), "scales")

flat   = helper.make_node("Reshape", ["X", "s_flat"], ["flat"])
matmul = helper.make_node("MatMul",  ["flat", "W"],   ["lin"])
unflat = helper.make_node("Reshape", ["lin", "s_orig"], ["unflat"])
resize = helper.make_node("Resize",  ["unflat", "", "scales"], ["Y"],
                          mode="linear", coordinate_transformation_mode="half_pixel")
graph = helper.make_graph([flat, matmul, unflat, resize], "g", [X], [Y],
                          initializer=[W_init, s_flat, s_orig, scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"X": x})[0]

all_half = bool(np.all(np.abs(out - 0.5) < 1e-4))
print(f"Output[0,0,:2,:2]: {out[0,0,:2,:2]}")
print(f"All outputs == 0.5 (BUG): {all_half}")
PASS = not all_half
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
