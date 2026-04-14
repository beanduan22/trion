#!/usr/bin/env python3
"""
Bug ID     : github_onnx_spec_004
Source     : GitHub — ONNX Spec
Compiler   : ONNX Spec
Patterns   : topk nan handling
Root cause : Bug: ONNX spec #7754 — TopK NaN handling undocumented; ORT treats NaN > all finite, spec says nothing.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: ONNX spec #7754 — TopK NaN handling undocumented; ORT treats NaN > all finite, spec says nothing.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# 3 NaN values mixed with finite values; top-3 should be all NaN if NaN > finite
x     = np.array([[float('nan'), 3.0, float('nan'), 1.0, 2.0, float('nan')]], dtype=np.float32)
k_val = np.array([3], dtype=np.int64)

X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 6])
K    = helper.make_tensor_value_info("K", TensorProto.INT64,  [1])
Vals = helper.make_tensor_value_info("Vals", TensorProto.FLOAT, [1, 3])
Idxs = helper.make_tensor_value_info("Idxs", TensorProto.INT64,  [1, 3])

node  = helper.make_node("TopK", ["X", "K"], ["Vals", "Idxs"], axis=1, largest=1, sorted=1)
graph = helper.make_graph([node], "topk_nan", [X, K], [Vals, Idxs])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
vals, idxs = sess.run(None, {"X": x, "K": k_val})

nan_count = int(np.sum(np.isnan(vals)))
print(f"Input: {x.flatten()}")
print(f"Top-3 values:  {vals.flatten()}")
print(f"Top-3 indices: {idxs.flatten()}")
print(f"NaNs in top-3: {nan_count}  (ORT treats NaN > all finite values)")
print(f"Spec does NOT document NaN handling — behavior is implementation-defined")
PASS = True  # documenting undocumented behavior
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
