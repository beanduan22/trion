#!/usr/bin/env python3
"""
Bug ID     : github_openvino_008
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : batchnorm decomposition type error
Root cause : OpenVINO Bug #23539 - Shape inference error on Multiply in BatchNorm decomposition
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO Bug #23539 - Shape inference error on Multiply in BatchNorm decomposition
# https://github.com/openvinotoolkit/openvino/issues/23539
# OV bug: type mismatch on Multiply node during BatchNorm1d decomposition in Conv+BN fusion
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(9)
N, C = 4, 8
x     = np.random.randn(N, C).astype(np.float32)
gamma = np.ones(C,  dtype=np.float32)
beta  = np.zeros(C, dtype=np.float32)
mean  = np.zeros(C, dtype=np.float32)
var   = np.ones(C,  dtype=np.float32)

X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C])
Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, C])
inits = [
    numpy_helper.from_array(gamma, "scale"),
    numpy_helper.from_array(beta,  "B"),
    numpy_helper.from_array(mean,  "mean"),
    numpy_helper.from_array(var,   "var"),
]
node  = helper.make_node("BatchNormalization",
                         ["X", "scale", "B", "mean", "var"], ["Y"],
                         epsilon=1e-5, training_mode=0)
graph = helper.make_graph([node], "g", [X], [Y], initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

expected = (x - mean) / np.sqrt(var + 1e-5) * gamma + beta
max_diff = float(np.max(np.abs(ort_out - expected)))
print(f"input shape: {x.shape}  (2D BatchNorm1d pattern)")
print(f"ort_out[0,:4]:  {ort_out[0, :4]}")
print(f"expected[0,:4]: {expected[0, :4]}")
print(f"max_diff: {max_diff:.2e}")
print(f"PASS={max_diff < 1e-4}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
