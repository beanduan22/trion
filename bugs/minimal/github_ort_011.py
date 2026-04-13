#!/usr/bin/env python3
"""
Bug ID     : github_ort_011
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : convtranspose autopad same upper
Root cause : Bug: ORT ConvTranspose auto_pad=SAME_UPPER ignored attribute, used explicit pads=[0,0,0,0] (issue #4086).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Bug: ORT ConvTranspose auto_pad=SAME_UPPER ignored attribute, used explicit pads=[0,0,0,0] (issue #4086).

np.random.seed(7)
x = np.random.randn(1, 1, 4, 4).astype(np.float32)
w = np.random.randn(1, 1, 3, 3).astype(np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 8])
W_init = numpy_helper.from_array(w, "W")

# ConvTranspose with auto_pad=SAME_UPPER, stride=2
# ONNX spec: output_shape = input_shape * strides = [1,1,8,8]
node = helper.make_node(
    "ConvTranspose",
    inputs=["X", "W"],
    outputs=["Y"],
    auto_pad="SAME_UPPER",
    strides=[2, 2],
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[W_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
onnx.checker.check_model(model)

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out = sess.run(None, {"X": x})[0]

expected_shape = (1, 1, 8, 8)
actual_shape = out.shape
shape_ok = actual_shape == expected_shape

print(f"Input shape: {x.shape}, stride=2, auto_pad=SAME_UPPER")
print(f"Expected output shape: {expected_shape}")
print(f"Actual  output shape:  {actual_shape}")
print(f"Shape correct: {shape_ok}")

# For SAME_UPPER stride=2: output = input*stride = 8×8 (verified above)
# Sanity check: output values should be finite
all_finite = bool(np.all(np.isfinite(out)))
print(f"All outputs finite: {all_finite}")
print(f"Output[0,0,:3,:3]: {out[0,0,:3,:3]}")
print(f"PASS={shape_ok and all_finite}")

PASS = shape_ok and all_finite
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
