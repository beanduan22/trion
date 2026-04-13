#!/usr/bin/env python3
"""
Bug ID     : github_onnx_spec_005
Source     : GitHub — ONNX Spec
Compiler   : ONNX Spec
Patterns   : topk tie breaking
Root cause : Bug: ONNX spec #3501 — TopK tie-breaking ambiguous; spec now requires lower index wins on equal values.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: ONNX spec #3501 — TopK tie-breaking ambiguous; spec now requires lower index wins on equal values.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Four equal maxima at indices 0,1,2,4; top-3 must be indices 0,1,2
x     = np.array([[5.0, 5.0, 5.0, 3.0, 5.0]], dtype=np.float32)
k_val = np.array([3], dtype=np.int64)

X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5])
K    = helper.make_tensor_value_info("K", TensorProto.INT64,  [1])
Vals = helper.make_tensor_value_info("Vals", TensorProto.FLOAT, [1, 3])
Idxs = helper.make_tensor_value_info("Idxs", TensorProto.INT64,  [1, 3])

node  = helper.make_node("TopK", ["X", "K"], ["Vals", "Idxs"], axis=1, largest=1, sorted=1)
graph = helper.make_graph([node], "topk_tie", [X, K], [Vals, Idxs])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
vals, idxs = sess.run(None, {"X": x, "K": k_val})

expected_idxs = np.array([[0, 1, 2]])
correct = np.array_equal(idxs, expected_idxs)
print(f"Input:          {x.flatten()}")
print(f"Top-3 indices:  {idxs.flatten()}  (expected [0,1,2] — lower index wins)")
PASS = correct
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
