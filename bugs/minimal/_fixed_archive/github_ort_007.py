#!/usr/bin/env python3
"""
Bug ID     : github_ort_007
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : topk gpu nondeterministic
Root cause : Input with many equal values — the tie-breaking case
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
"""
ORT Bug #3391 — GPU TopK non-deterministic on equal values (CUDA bitonic sort race)
https://github.com/microsoft/onnxruntime/issues/3391
Status: Closed/Fixed (PR #3758)

Root cause: CUDA bitonic sort had a thread-sync race condition. When multiple
elements have equal float values, the order is non-deterministic across runs.
On CPU this is deterministic. This repro shows CPU behavior and documents the
expected stable sort semantics (lower index wins on tie).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Input with many equal values — the tie-breaking case
x = np.array([[3.0, 3.0, 3.0, 1.0, 3.0, 2.0, 3.0, 3.0]], dtype=np.float32)
k_val = np.array([3], dtype=np.int64)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
K = helper.make_tensor_value_info("K", TensorProto.INT64,  [1])
Vals = helper.make_tensor_value_info("Vals",  TensorProto.FLOAT, [1, 3])
Idxs = helper.make_tensor_value_info("Idxs",  TensorProto.INT64,  [1, 3])

node = helper.make_node("TopK", ["X", "K"], ["Vals", "Idxs"], axis=1, largest=1, sorted=1)
graph = helper.make_graph([node], "topk", [X, K], [Vals, Idxs])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
results = []
for _ in range(10):
    vals, idxs = sess.run(None, {"X": x, "K": k_val})
    results.append(tuple(idxs.flatten().tolist()))

unique = set(results)
print(f"Input:   {x.flatten()}")
print(f"Top-3 indices over 10 runs: {unique}")
print(f"All runs identical (deterministic on CPU): {len(unique) == 1}")
print(f"Expected lower indices first on tie: indices should be [0,1,2]")
print(f"PASS={len(unique) == 1}")

PASS = len(unique) == 1
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
