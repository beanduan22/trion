#!/usr/bin/env python3
"""
Bug ID     : github_inductor_011
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : scatternd oob indices
Root cause : Bug: Inductor #122291 — ScatterND OOB indices returned garbage on GPU; eager raised IndexError correctly.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: Inductor #122291 — ScatterND OOB indices returned garbage on GPU; eager raised IndexError correctly.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

data    = np.arange(12, dtype=np.float32).reshape(3, 4)
indices = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)
updates = np.array([100.0, 200.0, 300.0], dtype=np.float32)

D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [3, 4])
I = helper.make_tensor_value_info("I", TensorProto.INT64,  [3, 2])
U = helper.make_tensor_value_info("U", TensorProto.FLOAT,  [3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT,  [3, 4])

node  = helper.make_node("ScatterND", ["D", "I", "U"], ["Y"])
graph = helper.make_graph([node], "g", [D, I, U], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out   = sess.run(None, {"D": data, "I": indices, "U": updates})[0]

expected = data.copy()
for idx, upd in zip(indices, updates):
    expected[tuple(idx)] = upd

max_diff = float(np.max(np.abs(out - expected)))
print(f"Max diff vs expected: {max_diff:.6f}")
PASS = max_diff < 1e-6
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
