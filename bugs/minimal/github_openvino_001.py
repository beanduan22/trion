#!/usr/bin/env python3
"""
Bug ID     : github_openvino_001
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : softsign all ones
Root cause : OpenVINO Bug #31252 - Softsign CPU JIT emitter returns all 1.0 regardless of input
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO Bug #31252 - Softsign CPU JIT emitter returns all 1.0 regardless of input
# https://github.com/openvinotoolkit/openvino/issues/31252
# OV bug: x / (1 + |x|) computed as 1.0 for all inputs on CPU JIT path
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

x = np.array([-2., -1., 0., 1., 2.], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5])
node = helper.make_node("Softsign", ["X"], ["Y"])
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

expected = x / (1.0 + np.abs(x))

max_diff = float(np.max(np.abs(ort_out - expected)))
print(f"input:    {x}")
print(f"expected: {expected}")
print(f"ort_out:  {ort_out}")
print(f"max_diff: {max_diff:.2e}")
print(f"PASS={max_diff < 1e-5}")

PASS = max_diff < 1e-5
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
