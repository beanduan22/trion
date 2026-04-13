#!/usr/bin/env python3
"""
Cross-compiler bug: Resize nearest — wrong pixel selection across coordinate modes
==================================================================================
Compilers affected : OnnxRuntime (bugs #14407, #7982), TVM (PR #10401), TensorFlow (#57780, MLIR/TOSA)
Shared root cause  : Each compiler implements one or more nearest-neighbour coordinate
                     modes incorrectly, producing off-by-one pixel selections:
                     - ORT #14407: tf_half_pixel_for_nn applied to batch/channel dims (wrong)
                     - ORT #7982:  nearest 1→64 one pixel off vs PyTorch
                     - TVM PR#10401: topi validation blocked nearest+align_corners+round_prefer_ceil
                     - TF #57780:  TFLite converter forced align_corners=False instead of True
                     - TF MLIR/TOSA: MLIR resize shifted rows by 1 with half_pixel_centers=True
Status             : Bugs closed/fixed; repro documents the patterns and validates ORT CPU.

Three sub-tests:
  A) tf_half_pixel_for_nn: must apply only to spatial dims (not N/C).
  B) align_corners 1×1→4×4: formula gives x_orig = 0.0 for all outputs.
  C) pytorch_half_pixel 1×1→1×1: formula gives x_orig = -0.5, should clip to 0.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


def run_resize(src, out_h, out_w, coord_mode, nearest_mode="round_prefer_floor",
               extra_attrs=None):
    H, W = src.shape[2], src.shape[3]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(src.shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [src.shape[0], src.shape[1], out_h, out_w])
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4],
                                [1.0, 1.0, out_h / H, out_w / W])
    attrs = dict(
        mode="nearest",
        coordinate_transformation_mode=coord_mode,
        nearest_mode=nearest_mode,
    )
    if extra_attrs:
        attrs.update(extra_attrs)
    node  = helper.make_node("Resize", ["X", "", "scales"], ["Y"], **attrs)
    graph = helper.make_graph([node], "g", [X], [Y], initializer=[scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"X": src})[0]


np.random.seed(7)
src_small = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)

print("=== gh_bug_002: Resize nearest coordinate modes — ORT + TVM + TF ===")

# ── Test A: tf_half_pixel_for_nn (ORT #14407) ────────────────────────────────
# Must only transform spatial dims; batch/channel axes must map 1:1.
out_A = run_resize(src_small, out_h=6, out_w=6, coord_mode="tf_half_pixel_for_nn")
shape_ok_A = out_A.shape == (1, 1, 6, 6)
finite_A   = bool(np.all(np.isfinite(out_A)))
print(f"A) tf_half_pixel_for_nn 3x3→6x6: shape={out_A.shape} (expected (1,1,6,6))")
print(f"   first row: {out_A[0,0,0,:6]}")
print(f"   shape_ok={shape_ok_A}, finite={finite_A}")

# ── Test B: align_corners with round_prefer_ceil (TVM PR#10401) ───────────────
# TVM validation blocked this combo; ORT must handle it correctly.
src_1x1 = np.array([[[[42.0]]]], dtype=np.float32)
out_B   = run_resize(src_1x1, out_h=4, out_w=4, coord_mode="align_corners",
                     nearest_mode="round_prefer_ceil")
shape_ok_B = out_B.shape == (1, 1, 4, 4)
# align_corners 1→4: all outputs map back to the single source pixel → all = 42
all_42_B = bool(np.allclose(out_B, 42.0))
print(f"B) align_corners + round_prefer_ceil, 1x1→4x4: all=42? {all_42_B}")
print(f"   out_B: {out_B.flatten()}")

# ── Test C: pytorch_half_pixel, output_size=1 (ONNX spec #5080) ───────────────
# x_orig = (0 + 0.5) / 1.0 - 0.5 = 0.0 → should select pixel 0.
out_C   = run_resize(src_1x1, out_h=1, out_w=1, coord_mode="pytorch_half_pixel")
shape_ok_C = out_C.shape == (1, 1, 1, 1)
correct_C  = bool(np.allclose(out_C, 42.0))
print(f"C) pytorch_half_pixel, 1x1→1x1: value={float(out_C.flat[0]):.4f} (expected 42.0)")

# ── Test D: nearest 1D resize, half_pixel, off-by-one (ORT #7982 pattern) ─────
src_1d = np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4)
out_D  = run_resize(src_1d, out_h=1, out_w=8, coord_mode="half_pixel")
# half_pixel nearest for 4→8: x_orig = (i+0.5)/2 - 0.5 → round to [0,0,1,1,2,2,3,3]
expected_D = np.array([[[[0., 0., 1., 1., 2., 2., 3., 3.]]]], dtype=np.float32)
ok_D = bool(np.allclose(out_D, expected_D))
print(f"D) half_pixel nearest 4→8: {out_D.flatten()} (expected {expected_D.flatten()})")
print(f"   correct={ok_D}")

PASS = shape_ok_A and finite_A and shape_ok_B and all_42_B and shape_ok_C and correct_C and ok_D
print(f"Bugs: ORT #14407/#7982, TVM PR#10401, TF #57780/MLIR")
print(f"PASS={PASS}")
