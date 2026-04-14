#!/usr/bin/env python3
"""
Bug ID     : github_openvino_012
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : prelu arm emitter
Root cause : OpenVINO PR#28223 - PReLU ARM JIT emitter applies wrong slope for negative inputs
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO PR#28223 - PReLU ARM JIT emitter applies wrong slope for negative inputs
# https://github.com/openvinotoolkit/openvino/pull/28223
# OV bug: ARM SVE/NEON emitter used output register instead of slope register,
# computing x*x for negatives instead of slope*x
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

x     = np.array([-3., -2., -1., 0., 1., 2.], dtype=np.float32)
slope = np.array([0.25], dtype=np.float32)

X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [6])
Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [6])
S    = numpy_helper.from_array(slope, "slope")
node = helper.make_node("PRelu", ["X", "slope"], ["Y"])
graph = helper.make_graph([node], "g", [X], [Y], initializer=[S])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

# PReLU: x if x>=0 else slope*x
expected = np.where(x >= 0, x, slope[0] * x)
# ARM bug: computes x*x for negatives -> [-9, -4, -1, 0, 1, 2]
arm_bug  = np.where(x >= 0, x, x * x)

max_diff = float(np.max(np.abs(ort_out - expected)))
print(f"input:    {x}")
print(f"ort_out:  {ort_out}")
print(f"expected: {expected}  (slope*x for negatives)")
print(f"arm_bug:  {arm_bug}  (x*x for negatives)")
print(f"max_diff: {max_diff:.2e}")
print(f"PASS={max_diff < 1e-5}")

PASS = max_diff < 1e-5
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
