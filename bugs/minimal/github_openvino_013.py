#!/usr/bin/env python3
"""
Bug ID     : github_openvino_013
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : convtranspose autopad output shape
Root cause : OpenVINO Bug #30798 - ConvTranspose auto_pad=SAME_LOWER gives wrong output shape
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO Bug #30798 - ConvTranspose auto_pad=SAME_LOWER gives wrong output shape
# https://github.com/openvinotoolkit/openvino/issues/30798
# OV bug: SAME_LOWER used SAME_UPPER padding formula, causing shape mismatch
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(89)
x = np.random.randn(1, 1, 4, 4).astype(np.float32)
w = np.random.randn(1, 1, 3, 3).astype(np.float32)

X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
W    = numpy_helper.from_array(w, "W")
node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                        auto_pad="SAME_LOWER", strides=[2, 2])
graph = helper.make_graph([node], "g", [X], [Y], initializer=[W])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"X": x})[0]

# SAME_LOWER stride=2: output = input * stride = 4*2 = 8 -> [1,1,8,8]
expected_shape = (1, 1, 8, 8)
shape_ok = out.shape == expected_shape
print(f"input shape: {x.shape}, kernel=3x3, stride=2, auto_pad=SAME_LOWER")
print(f"output shape: {out.shape}  (expected {expected_shape})")
print(f"OV bug: SAME_LOWER applied SAME_UPPER formula, causing 1-pixel shape error")
print(f"PASS={shape_ok}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
