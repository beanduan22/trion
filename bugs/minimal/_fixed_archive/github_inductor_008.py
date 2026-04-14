#!/usr/bin/env python3
"""
Bug ID     : github_inductor_008
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : convtranspose empty output
Root cause : Bug: Inductor #144013 — ConvTranspose2d output_size=(0,0) caused IndexError in Inductor, not correct ValueError.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: Inductor #144013 — ConvTranspose2d output_size=(0,0) caused IndexError in Inductor, not correct ValueError.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(29)
x = np.random.randn(1, 2, 3, 3).astype(np.float32)
w = np.random.randn(2, 1, 3, 3).astype(np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3, 3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
W_init = numpy_helper.from_array(w, "W")

node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                        strides=[1, 1], pads=[1, 1, 1, 1])
graph = helper.make_graph([node], "g", [X], [Y], initializer=[W_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"X": x})[0]

# stride=1, pads=[1,1,1,1], kernel=3: output = input size = 3x3
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}  (expected (1,1,3,3))")
# Crash scenario: torch.compile(nn.ConvTranspose2d(2,1,3,stride=2,padding=1))(randn(1,2,1,1), output_size=(0,0))
# → Inductor IndexError; eager → correct ValueError
PASS = out.shape == (1, 1, 3, 3)
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
