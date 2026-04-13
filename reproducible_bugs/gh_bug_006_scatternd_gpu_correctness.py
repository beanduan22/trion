#!/usr/bin/env python3
"""
Cross-compiler bug: ScatterND — GPU race conditions and OOB index handling
==========================================================================
Compilers affected : OnnxRuntime (PR #23755), TVM (PR #7447), PyTorch Inductor (#122291)
Shared root cause  : All three compilers have GPU-specific ScatterND bugs:
                     - ORT PR#23755:  JSEP/WebGPU: race with duplicate indices → non-deterministic
                     - TVM PR#7447:   CUDA kernel missing 'return' → thread continues → memory corruption
                     - Inductor #122291: OOB indices silently returned garbage on GPU;
                                         eager correctly raised IndexError
Status             : All bugs closed/fixed. CPU is always correct in all three frameworks.

Three sub-tests:
  A) Duplicate indices (ORT PR#23755): CPU last-write-wins semantics.
  B) Diagonal scatter (TVM PR#7447): correct values on 4×4 zeros.
  C) In-bounds scatter (Inductor #122291): in-bounds indices must give exact results.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def make_scatternd(data, indices, updates, opset=16, reduction="none"):
    nd = data.ndim
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, list(data.shape))
    I = helper.make_tensor_value_info("I", TensorProto.INT64, list(indices.shape))
    U = helper.make_tensor_value_info("U", TensorProto.FLOAT, list(updates.shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(data.shape))
    attrs = {}
    if opset >= 16:
        attrs["reduction"] = reduction
    node  = helper.make_node("ScatterND", ["D","I","U"], ["Y"], **attrs)
    graph = helper.make_graph([node], "g", [D, I, U], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"D": data, "I": indices, "U": updates})[0]


print("=== gh_bug_006: ScatterND GPU correctness — ORT + TVM + Inductor ===")

# ── Test A: Duplicate indices — CPU last-write-wins (ORT PR#23755) ─────────────
data_A    = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
indices_A = np.array([[2], [2], [4]], dtype=np.int64)   # index 2 duplicated
updates_A = np.array([10.0, 20.0, 30.0], dtype=np.float32)
out_A     = make_scatternd(data_A, indices_A, updates_A)
# CPU sequential: last write (update=20.0) wins at index 2
expected_A = np.array([1.0, 2.0, 20.0, 4.0, 30.0], dtype=np.float32)
ok_A = float(np.max(np.abs(out_A - expected_A))) < 1e-5
print(f"A) Duplicate indices: out={out_A}  expected={expected_A}  ok={ok_A}")
print(f"   ORT PR#23755: JSEP/WebGPU race — duplicate-index result non-deterministic on GPU")

# ── Test B: Diagonal scatter (TVM PR#7447 — missing 'return' in CUDA kernel) ──
data_B    = np.zeros((4, 4), dtype=np.float32)
indices_B = np.array([[0,0],[1,1],[2,2],[3,3]], dtype=np.int64)
updates_B = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
out_B     = make_scatternd(data_B, indices_B, updates_B, opset=13)
expected_B = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
ok_B = float(np.max(np.abs(out_B - expected_B))) < 1e-6
print(f"B) Diagonal scatter: ok={ok_B}")
print(f"   TVM PR#7447: CUDA kernel missing 'return' → continued past update → corruption")
print(f"   out_B:\n{out_B}")

# ── Test C: In-bounds 2D scatter (Inductor #122291 pattern) ────────────────────
data_C    = np.arange(12, dtype=np.float32).reshape(3, 4)
indices_C = np.array([[0,0],[1,2],[2,3]], dtype=np.int64)
updates_C = np.array([100.0, 200.0, 300.0], dtype=np.float32)
out_C     = make_scatternd(data_C, indices_C, updates_C, opset=13)
expected_C = data_C.copy()
for idx, upd in zip(indices_C, updates_C):
    expected_C[tuple(idx)] = upd
ok_C = float(np.max(np.abs(out_C - expected_C))) < 1e-6
print(f"C) In-bounds 2D scatter: ok={ok_C}")
print(f"   Inductor #122291: GPU returned garbage for OOB indices; CPU raises IndexError")

# ── Test D: ScatterND with reduction='add' ────────────────────────────────────
data_D    = np.zeros(5, dtype=np.float32)
indices_D = np.array([[1],[1],[3]], dtype=np.int64)
updates_D = np.array([2.0, 3.0, 5.0], dtype=np.float32)
out_D     = make_scatternd(data_D, indices_D, updates_D, opset=16, reduction="add")
expected_D = np.array([0., 5., 0., 5., 0.], dtype=np.float32)  # 2+3=5 at index 1
ok_D = float(np.max(np.abs(out_D - expected_D))) < 1e-5
print(f"D) Reduction='add' duplicates: out={out_D}  expected={expected_D}  ok={ok_D}")

PASS = ok_A and ok_B and ok_C and ok_D
print(f"Bugs: ORT PR#23755, TVM PR#7447, Inductor #122291")
print(f"PASS={PASS}")
