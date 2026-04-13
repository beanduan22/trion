#!/usr/bin/env python3
"""
Bug ID     : github_openvino_005
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : mul zero nan hazard
Root cause : OpenVINO Bug #8729 - mul_zero_elim: x * 0 -> 0 is unsound when x = NaN
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO Bug #8729 - mul_zero_elim: x * 0 -> 0 is unsound when x = NaN
# https://github.com/openvinotoolkit/openvino/issues/8729
# OV bug: optimizer folded x*0 to constant 0, but IEEE 754 requires NaN*0 = NaN
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

x    = np.array([float('nan'), float('inf'), 1.0], dtype=np.float32)
zero = np.zeros(3, dtype=np.float32)

Z = numpy_helper.from_array(zero, "zeros")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
node  = helper.make_node("Mul", ["X", "zeros"], ["Y"])
graph = helper.make_graph([node], "g", [X], [Y], initializer=[Z])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# No-opt session: computes x*0 at runtime (correct IEEE 754)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
out = sess.run(None, {"X": x})[0]

# IEEE 754: NaN*0=NaN, inf*0=NaN, 1*0=0
expected = x * zero

print(f"input:    {x}")
print(f"ort_out:  {out}")
print(f"ieee754:  {expected}")
print(f"NaN preserved: {np.isnan(out[0])}  (OV bug: optimizer folded to 0.0)")
print(f"inf*0 is NaN:  {np.isnan(out[1])}")
print(f"PASS={bool(np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 0.0)}")

PASS = bool(np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 0.0)
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
