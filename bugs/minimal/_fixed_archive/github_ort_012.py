#!/usr/bin/env python3
"""
Bug ID     : github_ort_012
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : convtranspose dilation outputpadding
Root cause : Bug: ORT ConvTranspose dilation>1 + output_padding gives wrong output values (issue #14208).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn

# Bug: ORT ConvTranspose dilation>1 + output_padding gives wrong output values (issue #14208).
np.random.seed(13)
x_np = np.random.randn(1, 2, 5, 5).astype(np.float32)
w_np = np.random.randn(2, 1, 3, 3).astype(np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 5, 5])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
W_init = numpy_helper.from_array(w_np, "W")

node = helper.make_node(
    "ConvTranspose",
    inputs=["X", "W"],
    outputs=["Y"],
    dilations=[2, 2],
    strides=[2, 2],
    output_padding=[1, 1],
    pads=[0, 0, 0, 0],
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[W_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x_np})[0]

conv_t = nn.ConvTranspose2d(2, 1, 3, stride=2, padding=0, output_padding=1, dilation=2, bias=False)
with torch.no_grad():
    conv_t.weight.copy_(torch.from_numpy(w_np))
torch_out = conv_t(torch.from_numpy(x_np)).detach().numpy()

shape_match = ort_out.shape == torch_out.shape
max_diff = float(np.max(np.abs(ort_out - torch_out))) if shape_match else float("inf")
print(f"ORT shape: {ort_out.shape}, Torch shape: {torch_out.shape}")
print(f"ORT output[0,0,0,:5]:   {ort_out[0,0,0,:5]}")
print(f"Torch output[0,0,0,:5]: {torch_out[0,0,0,:5]}")
print(f"Max diff: {max_diff:.6f}")
PASS = shape_match and max_diff < 1e-4
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
