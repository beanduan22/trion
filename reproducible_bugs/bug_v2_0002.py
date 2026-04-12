#!/usr/bin/env python3
"""
Bug v2-0002 — ORT Resize(nearest, ceil, half_pixel) + Cast float->int->float roundtrip
Compiler  : OnnxRuntime (ORT_ENABLE_ALL diverges from pytorch_eager)
Root cause: ONNX Resize with mode='nearest', nearest_mode='ceil',
            coordinate_transformation_mode='half_pixel' uses
              x_src = (x_dst + 0.5) / scale - 0.5
            then applies ceil() to pick the source pixel.
            pytorch's equivalent uses floor() by default, causing different pixels
            to be selected at upsampling boundaries.
            The downstream Cast(float->int32)->Cast(int32->float) amplifies
            fractional residuals into discrete steps.
Status    : Active (ORT_DISABLE_ALL == ORT_ENABLE_ALL; divergence is vs pytorch)

Minimal trigger: Resize(nearest, ceil, half_pixel) on a small float tensor.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(2)

# ── Build minimal ONNX model ──────────────────────────────────────────────────
C, H, W = 2, 4, 4

X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, C, H * 2, W * 2])

scales_init = numpy_helper.from_array(
    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name='scales')
roi_init = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')

# Resize nearest with ceil rounding and half_pixel coordinate transform
resize_node = helper.make_node(
    'Resize',
    inputs=['X', 'roi', 'scales'],
    outputs=['resize_out'],
    coordinate_transformation_mode='half_pixel',
    mode='nearest',
    nearest_mode='ceil',
)
# Cast float->int32->float (quantization roundtrip that exposes boundary differences)
cast_i32 = helper.make_node('Cast', ['resize_out'], ['cast_i32'], to=int(TensorProto.INT32))
cast_f32 = helper.make_node('Cast', ['cast_i32'],   ['Y'],        to=int(TensorProto.FLOAT))

graph = helper.make_graph(
    [resize_node, cast_i32, cast_f32],
    'bug_v2_0002',
    [X_info], [Y_info],
    initializer=[scales_init, roi_init],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 7
model_bytes = model.SerializeToString()

# Use small integer values so Cast behavior is deterministic
INPUT = np.arange(1.0, 1.0 + C * H * W, dtype=np.float32).reshape(1, C, H, W)

# ── ORT (opt enabled) ─────────────────────────────────────────────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

# ── ORT (opt disabled) ────────────────────────────────────────────────────────
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

max_diff = float(np.max(np.abs(got - expected)))

# ── Show the coordinate mode semantics difference ─────────────────────────────
# Compare: half_pixel+ceil vs floor (pytorch default) for nearest resize
def nearest_resize_manual(x, scale_h, scale_w, ctm='half_pixel', rounding='ceil'):
    """Manual nearest-neighbor resize."""
    N, C, H, W = x.shape
    out_H, out_W = int(H * scale_h), int(W * scale_w)
    out = np.zeros((N, C, out_H, out_W), dtype=x.dtype)
    for i in range(out_H):
        for j in range(out_W):
            if ctm == 'half_pixel':
                y_src = (i + 0.5) / scale_h - 0.5
                x_src = (j + 0.5) / scale_w - 0.5
            else:  # asymmetric
                y_src = i / scale_h
                x_src = j / scale_w
            if rounding == 'ceil':
                yi = min(int(np.ceil(y_src)), H - 1)
                xi = min(int(np.ceil(x_src)), W - 1)
            elif rounding == 'round_prefer_floor':
                yi = min(int(np.round(y_src - 0.5 * (y_src == int(y_src + 0.5)))), H - 1)
                xi = min(int(np.round(x_src - 0.5 * (x_src == int(x_src + 0.5)))), W - 1)
            else:  # floor
                yi = min(max(int(np.floor(y_src)), 0), H - 1)
                xi = min(max(int(np.floor(x_src)), 0), W - 1)
            out[:, :, i, j] = x[:, :, max(yi, 0), max(xi, 0)]
    return out

# Use tiny input to show pixel selection difference
x_small = np.arange(1.0, 5.0, dtype=np.float32).reshape(1, 1, 2, 2)
ceil_out   = nearest_resize_manual(x_small, 2, 2, 'half_pixel', 'ceil')
floor_out  = nearest_resize_manual(x_small, 2, 2, 'half_pixel', 'floor')

print("=== Bug v2-0002: Resize(nearest, ceil, half_pixel) pixel selection ===")
print(f"Input 2x2:\n{x_small[0,0]}")
print(f"\nhalf_pixel + ceil (ONNX/ORT) output 4x4:\n{ceil_out[0,0]}")
print(f"\nhalf_pixel + floor (pytorch default) output 4x4:\n{floor_out[0,0]}")
ceil_vs_floor = float(np.max(np.abs(ceil_out - floor_out)))
print(f"\nceil vs floor max_diff: {ceil_vs_floor:.4f}  (different pixels selected!)")

print(f"\nFull model: ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff: {max_diff:.4e}")
print(f"expected[:4]: {expected.ravel()[:4]}")
print(f"got[:4]:      {got.ravel()[:4]}")

pass_flag = max_diff < 1e-4
print(f"\nPASS={pass_flag}")
if pass_flag:
    print("ORT_ENABLE_ALL == ORT_DISABLE_ALL (no internal ORT optimizer bug).")
    print("Divergence from pytorch is due to different nearest_mode: ceil vs floor.")
print(f"Coordinate-mode semantic diff (ceil vs floor half_pixel): {ceil_vs_floor:.4f}")
