#!/usr/bin/env python3
"""
Cross-compiler bug: InstanceNormalization — FP16 accumulation and small spatial dims
=====================================================================================
Compilers affected : OnnxRuntime (PR #9879), TVM (#15683)
Shared root cause  : Both compilers compute wrong InstanceNorm output in different
                     precision scenarios:
                     - ORT PR#9879:  GPU FP16 kernel accumulated mean/variance in FP16,
                                     causing overflow/rounding for large-variance inputs (std≈50)
                     - TVM #15683:   GPU parallel reduction computed wrong mean/variance
                                     for very small spatial dimensions (2×2 = 4 elements)
                     Both bugs only manifested on GPU; CPU is correct in both frameworks.
Status             : Both bugs closed/fixed. This repro validates CPU ORT as reference
                     and documents the input patterns that trigger each failure.

Three sub-tests:
  A) Large-variance FP16 vs FP32: significant difference expected (FP16 loses precision).
  B) Small spatial dims (2×2): ORT CPU must match numpy reference exactly.
  C) Typical usage (FP32, normal variance, 4×4 spatial): must be near-exact.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def run_instance_norm(x_np, dtype=TensorProto.FLOAT):
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    N, C, H, W = x_np.shape
    scale = np.ones(C,  dtype=np_dtype)
    bias  = np.zeros(C, dtype=np_dtype)
    X = helper.make_tensor_value_info("X",     dtype, [N, C, H, W])
    Y = helper.make_tensor_value_info("Y",     dtype, [N, C, H, W])
    S = numpy_helper.from_array(scale, "scale")
    B = numpy_helper.from_array(bias,  "bias")
    node  = helper.make_node("InstanceNormalization", ["X","scale","bias"], ["Y"], epsilon=1e-5)
    graph = helper.make_graph([node], "g", [X], [Y], initializer=[S, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 6)])
    sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out   = sess.run(None, {"X": x_np.astype(np_dtype)})[0]
    return out.astype(np.float32)


def numpy_instance_norm(x):
    """Reference: normalize each (n,c) slice over spatial dims."""
    out = np.zeros_like(x, dtype=np.float32)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            v   = x[n, c].astype(np.float32)
            mu  = v.mean()
            var = v.var()
            out[n, c] = (v - mu) / np.sqrt(var + 1e-5)
    return out


print("=== gh_bug_010: InstanceNorm precision — ORT PR#9879 + TVM #15683 ===")

# ── Test A: FP16 vs FP32 — large variance (ORT PR#9879 pattern) ───────────────
np.random.seed(41)
x_large = (np.random.randn(1, 4, 8, 8) * 50).astype(np.float32)  # std≈50 → overflow in FP16
out_fp32 = run_instance_norm(x_large, TensorProto.FLOAT)
out_fp16 = run_instance_norm(x_large, TensorProto.FLOAT16)
ref_A    = numpy_instance_norm(x_large)
diff_fp32_vs_ref = float(np.max(np.abs(out_fp32 - ref_A)))
diff_fp16_vs_ref = float(np.max(np.abs(out_fp16 - ref_A)))
# FP32 must be close; FP16 may diverge but should not be wildly wrong (ORT CPU uses FP32 reduction)
ok_A_fp32 = diff_fp32_vs_ref < 1e-3
ok_A_fp16 = diff_fp16_vs_ref < 0.5   # FP16 rounding expected; > this means overflow
print(f"A) Large variance (std≈50): FP32 diff={diff_fp32_vs_ref:.2e}  FP16 diff={diff_fp16_vs_ref:.4f}")
print(f"   ORT PR#9879: GPU FP16 accumulated in FP16 → overflow for std≈50")
print(f"   fp32_ok={ok_A_fp32}, fp16_ok={ok_A_fp16}")

# ── Test B: Small spatial dims 2×2 (TVM #15683 pattern) ──────────────────────
np.random.seed(17)
x_small = np.random.randn(2, 4, 2, 2).astype(np.float32)  # 4 elements per instance
out_B   = run_instance_norm(x_small, TensorProto.FLOAT)
ref_B   = numpy_instance_norm(x_small)
diff_B  = float(np.max(np.abs(out_B - ref_B)))
ok_B    = diff_B < 1e-4
print(f"B) Small spatial 2×2 (4 elems): max_diff={diff_B:.2e}  ok={ok_B}")
print(f"   TVM #15683: GPU parallel reduction wrong mean/var for small spatial dims")

# ── Test C: Standard FP32, normal variance, 4×4 spatial ─────────────────────
np.random.seed(7)
x_std = np.random.randn(2, 8, 4, 4).astype(np.float32)
out_C = run_instance_norm(x_std, TensorProto.FLOAT)
ref_C = numpy_instance_norm(x_std)
diff_C = float(np.max(np.abs(out_C - ref_C)))
ok_C   = diff_C < 1e-4
print(f"C) Standard FP32 4×4: max_diff={diff_C:.2e}  ok={ok_C}  (sanity check)")

# ── Test D: Single spatial element (extreme case for both bugs) ───────────────
x_1x1 = np.random.randn(1, 3, 1, 1).astype(np.float32)
out_D  = run_instance_norm(x_1x1, TensorProto.FLOAT)
# With a single element: mean = x, var = 0, result = 0 (normalized)
expected_D = np.zeros_like(x_1x1)
diff_D  = float(np.max(np.abs(out_D - expected_D)))
ok_D    = diff_D < 1e-5
print(f"D) Single spatial element (1×1): diff={diff_D:.2e}  ok={ok_D}  (must give 0)")

PASS = ok_A_fp32 and ok_A_fp16 and ok_B and ok_C and ok_D
print(f"Bugs: ORT PR#9879 (FP16 GPU accumulation), TVM #15683 (small spatial GPU reduction)")
print(f"PASS={PASS}")
