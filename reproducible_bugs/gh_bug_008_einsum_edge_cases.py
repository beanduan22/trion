#!/usr/bin/env python3
"""
Cross-compiler bug: Einsum — uppercase labels, scalar inputs, batch matmul
===========================================================================
Compilers affected : OnnxRuntime (#4944), OpenVINO (PR #30189), PyTorch Inductor (#85224)
Shared root cause  : Three different Einsum edge cases each broke one compiler:
                     - ORT #4944:      uppercase labels like "BIJ,BJK->BIK" rejected
                     - OV PR#30189:    scalar (rank-0) input operand caused segfault/null-deref
                     - Inductor #85224: MPS (Metal) batch matmul wrong on first call
                                        (Metal command buffer not initialized)
Status             : All bugs closed/fixed.

Three sub-tests:
  A) Uppercase equation: "BIJ,BJK->BIK" must give same result as lowercase "bij,bjk->bik".
  B) Scalar input: equation "i,->i" (vector * scalar) must work correctly.
  C) Batch matmul correctness: "bij,bjk->bik" must match numpy.einsum reference.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(5)


def run_einsum(equation, inputs_np, input_shapes):
    infos = [helper.make_tensor_value_info(f"X{i}", TensorProto.FLOAT, list(s))
             for i, s in enumerate(input_shapes)]
    Y     = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node  = helper.make_node("Einsum", [f"X{i}" for i in range(len(inputs_np))], ["Y"],
                             equation=equation)
    graph = helper.make_graph([node], "g", infos, [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feed  = {f"X{i}": arr for i, arr in enumerate(inputs_np)}
    return sess.run(None, feed)[0]


print("=== gh_bug_008: Einsum edge cases — ORT + OV + Inductor ===")

# ── Test A: Uppercase labels (ORT #4944) ──────────────────────────────────────
A = np.random.randn(2, 3, 4).astype(np.float32)
B = np.random.randn(2, 4, 5).astype(np.float32)

out_lower = run_einsum("bij,bjk->bik", [A, B], [(2,3,4), (2,4,5)])
out_upper = run_einsum("BIJ,BJK->BIK", [A, B], [(2,3,4), (2,4,5)])
diff_A = float(np.max(np.abs(out_lower - out_upper)))
ok_A   = diff_A < 1e-5
print(f"A) Uppercase: 'BIJ,BJK->BIK' vs 'bij,bjk->bik'  max_diff={diff_A:.2e}  ok={ok_A}")
print(f"   ORT #4944: uppercase equation was rejected with an error")

# ── Test B: Scalar (rank-0) input (OV PR#30189) ───────────────────────────────
vec    = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
scalar = np.array(3.0, dtype=np.float32)   # rank-0
out_B  = run_einsum("i,->i", [vec, scalar], [(5,), ()])
ref_B  = vec * float(scalar)
diff_B = float(np.max(np.abs(out_B - ref_B)))
ok_B   = diff_B < 1e-5
print(f"B) Scalar input 'i,->i': out={out_B}  ref={ref_B}  diff={diff_B:.2e}  ok={ok_B}")
print(f"   OV PR#30189: scalar (rank-0) operand caused segfault in parser")

# ── Test C: Batch matmul correctness (Inductor #85224) ───────────────────────
np.random.seed(71)
C = np.random.randn(2, 3, 4).astype(np.float32)
D = np.random.randn(2, 4, 5).astype(np.float32)
out_C  = run_einsum("bij,bjk->bik", [C, D], [(2,3,4), (2,4,5)])
ref_C  = np.einsum("bij,bjk->bik", C, D)
diff_C = float(np.max(np.abs(out_C - ref_C)))
ok_C   = diff_C < 1e-4
print(f"C) Batch matmul 'bij,bjk->bik': max_diff={diff_C:.2e}  ok={ok_C}")
print(f"   Inductor #85224: MPS batch matmul wrong on first call (Metal buf not initialized)")

# ── Test D: Outer product (no shared index) ───────────────────────────────────
E = np.array([1., 2., 3.], dtype=np.float32)
F = np.array([4., 5.], dtype=np.float32)
out_D  = run_einsum("i,j->ij", [E, F], [(3,), (2,)])
ref_D  = np.outer(E, F)
diff_D = float(np.max(np.abs(out_D - ref_D)))
ok_D   = diff_D < 1e-6
print(f"D) Outer product 'i,j->ij': max_diff={diff_D:.2e}  ok={ok_D}")

PASS = ok_A and ok_B and ok_C and ok_D
print(f"Bugs: ORT #4944, OV PR#30189, Inductor #85224")
print(f"PASS={PASS}")
