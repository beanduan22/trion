#!/usr/bin/env python3
"""
Cross-compiler bug: PReLU — wrong slope broadcast axis
=======================================================
Compilers affected : OpenVINO (PR #28223), TVM (PR #7208)
Shared root cause  : Both compilers apply the per-channel slope to the wrong axis:
                     - OV PR#28223: ARM SVE/NEON emitter used output register instead of
                                    slope register → effectively computed x*x for negatives
                     - TVM PR#7208: NCHW PReLU hardcoded axis=1 (channel dim); NHWC layout
                                    had H at axis=1, so slope was broadcast over H not C
                     Both bugs produce wrong values for any negative input.
Status             : Both bugs closed/fixed. This repro validates ORT CPU which is correct.

Three sub-tests:
  A) Per-channel slopes, NCHW: each channel has distinct slope, verify negative outputs.
  B) Scalar slope (broadcast to all channels): slope=0.25 for all negatives.
  C) Simulate the ARM bug output (x*x) vs correct (slope*x) to show the gap.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def run_prelu(x_np, slope_np):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_np.shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x_np.shape))
    S = numpy_helper.from_array(slope_np, "slope")
    node  = helper.make_node("PRelu", ["X", "slope"], ["Y"])
    graph = helper.make_graph([node], "g", [X], [Y], initializer=[S])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"X": x_np})[0]


print("=== gh_bug_009: PReLU slope broadcast axis — OV + TVM ===")

# ── Test A: Per-channel NCHW (TVM PR#7208 pattern) ────────────────────────────
np.random.seed(23)
x_A     = np.random.randn(1, 4, 3, 3).astype(np.float32)
# Distinct slope per channel — in NHWC layout, axis=1 is H not C
slope_A = np.array([0.1, 0.2, 0.5, 0.9], dtype=np.float32).reshape(4, 1, 1)
out_A   = run_prelu(x_A, slope_A)

# Reference: PReLU(x) = x if x>=0, slope[c]*x otherwise
slope_bc = slope_A.reshape(1, 4, 1, 1)
ref_A    = np.where(x_A >= 0, x_A, slope_bc * x_A)
diff_A   = float(np.max(np.abs(out_A - ref_A)))
ok_A     = diff_A < 1e-5

# Simulate TVM NHWC bug: would use axis=1 (H) instead of axis=2 (C in NHWC)
# For NCHW input, equivalent wrong slope would be broadcast to H-dim not C
tvm_bug_slope = np.array([0.1, 0.2, 0.5], dtype=np.float32).reshape(1, 1, 3, 1)  # H=3
tvm_bug = np.where(x_A >= 0, x_A, tvm_bug_slope * x_A)
diff_bug = float(np.max(np.abs(tvm_bug - ref_A)))
print(f"A) Per-channel NCHW: diff ORT vs ref = {diff_A:.2e}  ok={ok_A}")
print(f"   TVM bug gap (axis=H vs axis=C): {diff_bug:.4f}")

# ── Test B: Scalar slope (OV PR#28223 pattern) ────────────────────────────────
x_B     = np.array([-3., -2., -1., 0., 1., 2.], dtype=np.float32)
slope_B = np.array([0.25], dtype=np.float32)
out_B   = run_prelu(x_B, slope_B)
ref_B   = np.where(x_B >= 0, x_B, 0.25 * x_B)
# ARM bug: used x * x (output reg) instead of x * slope
arm_bug = np.where(x_B >= 0, x_B, x_B * x_B)
diff_B  = float(np.max(np.abs(out_B - ref_B)))
ok_B    = diff_B < 1e-5
print(f"B) Scalar slope=0.25: ORT output={out_B}")
print(f"   correct:   {ref_B}")
print(f"   ARM bug:   {arm_bug}")
print(f"   diff={diff_B:.2e}  ok={ok_B}")

# ── Test C: All-positive inputs — no negative path, both bugs disappear ────────
x_C    = np.abs(np.random.randn(2, 3, 4).astype(np.float32))  # all positive
slope_C = np.array([0.1, 0.2, 0.3], dtype=np.float32).reshape(3, 1)
out_C   = run_prelu(x_C, slope_C)
ref_C   = x_C.copy()   # no negatives → identity
diff_C  = float(np.max(np.abs(out_C - ref_C)))
ok_C    = diff_C < 1e-6
print(f"C) All-positive: diff={diff_C:.2e}  ok={ok_C}  (PReLU=identity for positives)")

# ── Test D: Verify negative-dominated input has large ARM-bug gap ──────────────
x_D    = -np.abs(np.random.randn(10).astype(np.float32)) - 1.0   # all negative
slope_D = np.array([0.1], dtype=np.float32)
out_D   = run_prelu(x_D, slope_D)
ref_D   = slope_D[0] * x_D
arm_D   = x_D * x_D   # what ARM bug would give
ok_D    = float(np.max(np.abs(out_D - ref_D))) < 1e-5
arm_gap = float(np.max(np.abs(arm_D - ref_D)))
print(f"D) All-negative (ARM bug impact): correct vs ARM-bug gap = {arm_gap:.4f}  ok={ok_D}")

PASS = ok_A and ok_B and ok_C and ok_D
print(f"Bugs: OV PR#28223 (ARM emitter x*x), TVM PR#7208 (NHWC axis=H not C)")
print(f"PASS={PASS}")
