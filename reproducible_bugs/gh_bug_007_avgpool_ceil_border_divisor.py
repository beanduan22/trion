#!/usr/bin/env python3
"""
Cross-compiler bug: AveragePool ceil_mode — border window divided by wrong count
=================================================================================
Compilers affected : PyTorch Inductor (#100987), OpenVINO (#20815)
Shared root cause  : Both compilers divide border (partial) windows by the full kernel
                     area instead of the actual number of elements in the window.
                     - Inductor #100987: 2D ceil_mode, border 2×2 window divided by 9 not 4
                     - OV #20815: int8 ceil_mode 1D, border 2-elem window divided by 3 not 2
                     Both bugs only manifest with ceil_mode=1 and count_include_pad=0.
Status             : Both bugs closed/fixed.

Three sub-tests:
  A) 2D ceil_mode: bottom-right border window has 2×2 elems, not 3×3.
  B) 1D ceil_mode: rightmost border window has 2 elems, not 3.
  C) No-border case: interior windows always divide by kernel_area — must be unaffected.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


def run_avgpool(x, kernel, strides, ceil_mode=1, count_include_pad=0):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node  = helper.make_node("AveragePool", ["X"], ["Y"],
                             kernel_shape=kernel,
                             strides=strides,
                             ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)
    graph = helper.make_graph([node], "g", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"X": x})[0]


print("=== gh_bug_007: AveragePool ceil_mode border divisor — Inductor + OV ===")

# ── Test A: 2D ceil_mode (Inductor #100987) ────────────────────────────────────
# 6×6 input, kernel 3×3, stride 2, ceil_mode → output 3×3
# Bottom-right window clips to [4:6, 4:6]: 4 elements
x_A = np.ones((1, 1, 6, 6), dtype=np.float32)
x_A[0, 0, 4, 4] = 5.0;  x_A[0, 0, 4, 5] = 6.0
x_A[0, 0, 5, 4] = 5.0;  x_A[0, 0, 5, 5] = 6.0
out_A = run_avgpool(x_A, kernel=[3,3], strides=[2,2])
border_val = float(out_A[0, 0, 2, 2])
correct    = (5.0 + 6.0 + 5.0 + 6.0) / 4   # = 5.5  (4 actual elements)
wrong      = (5.0 + 6.0 + 5.0 + 6.0) / 9   # ≈ 2.44 (bug: full 3×3 kernel)
ok_A = abs(border_val - correct) < 0.1
print(f"A) 2D ceil_mode border: got={border_val:.4f}  correct={correct:.4f}  bug_val≈{wrong:.4f}")
print(f"   Inductor #100987: divided by 9 (full kernel) → {wrong:.4f} instead of {correct:.4f}")
print(f"   ok={ok_A}")

# ── Test B: 1D ceil_mode (OV #20815 pattern) ──────────────────────────────────
# [1,1,1,6], kernel [1,3], stride [1,2], ceil_mode → output [1,1,1,3]
# Windows: [0:3] → 4.0, [2:5] → 8.0, [4:6] (2 elems) → correct=11.0, bug≈7.33
x_B = np.array([[[[2., 4., 6., 8., 10., 12.]]]], dtype=np.float32)
out_B = run_avgpool(x_B, kernel=[1,3], strides=[1,2])
border_B  = float(out_B[0, 0, 0, 2])
correct_B = (10.0 + 12.0) / 2   # = 11.0
wrong_B   = (10.0 + 12.0) / 3   # ≈ 7.33 (OV bug)
ok_B = abs(border_B - correct_B) < 0.1
print(f"B) 1D ceil_mode border: got={border_B:.4f}  correct={correct_B:.4f}  bug_val≈{wrong_B:.4f}")
print(f"   OV #20815: int8 kernel divided by 3 (full kernel) → {wrong_B:.4f} instead of {correct_B:.4f}")
print(f"   ok={ok_B}")

# ── Test C: Interior-only pool — no border → must be correct in all compilers ──
# 4×4 input, kernel 2×2, stride 2, ceil_mode → all windows fully covered
x_C   = np.ones((1, 1, 4, 4), dtype=np.float32) * 3.0
out_C = run_avgpool(x_C, kernel=[2,2], strides=[2,2])
all_three = bool(np.allclose(out_C, 3.0))
ok_C = all_three
print(f"C) Interior-only (no border windows): all_correct={ok_C}  (sanity check)")

# ── Test D: count_include_pad=1 — pad is counted → border = full kernel ────────
# With count_include_pad=1, bug disappears because even padding is counted
x_D   = np.array([[[[2., 4., 6., 8., 10., 12.]]]], dtype=np.float32)
out_D1 = run_avgpool(x_D, kernel=[1,3], strides=[1,2], count_include_pad=1)
out_D0 = run_avgpool(x_D, kernel=[1,3], strides=[1,2], count_include_pad=0)
border_D1 = float(out_D1[0,0,0,2])
border_D0 = float(out_D0[0,0,0,2])
# count_include_pad=1: pad fills zeros, (10+12+0)/3 = 7.33
# count_include_pad=0: actual overlap only, (10+12)/2 = 11.0
ok_D = border_D0 > border_D1  # 11.0 > 7.33
print(f"D) count_include_pad: 0→{border_D0:.4f}, 1→{border_D1:.4f}  semantics_correct={ok_D}")

PASS = ok_A and ok_B and ok_C and ok_D
print(f"Bugs: Inductor #100987, OV #20815")
print(f"PASS={PASS}")
