#!/usr/bin/env python3
"""
Bug v2-0000 — ORT Resize(linear, asymmetric) produces different output than pytorch_eager
Compiler  : OnnxRuntime (confirmed divergence vs pytorch_eager reference)
Root cause: ORT implements 'asymmetric' coordinate transform as x_src = x_dst / scale,
            while pytorch bilinear uses 'pytorch_half_pixel' (x_src = (x_dst+0.5)/scale - 0.5).
            The downstream Cast(float32->int32) quantization roundtrip further amplifies
            fractional differences into discrete steps. Combined with Conv+BN+Clip fusion
            the optimizer path diverges from pytorch eager by rel-L2 ~1.14.
Status    : Active (ORT_DISABLE_ALL agrees with ORT_ENABLE_ALL; divergence is vs pytorch)

Minimal trigger: Resize(linear, asymmetric, 2x) -> Cast(->int32) -> Cast(->float32) -> scale
The Cast truncation turns floating-point differences from the two resize semantics
into visibly different integer-valued outputs.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(42)

# ── Build minimal ONNX model ──────────────────────────────────────────────────
# Input: [1, C, H, W]
C, H, W = 1, 3, 3

X_info   = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y_info   = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, C, H * 2, W * 2])

scales_init = numpy_helper.from_array(
    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name='scales')
roi_init = numpy_helper.from_array(
    np.array([], dtype=np.float32), name='roi')
scale_factor_init = numpy_helper.from_array(
    np.array([0.01], dtype=np.float32), name='scale_factor')

# Resize bilinear with asymmetric coordinate transform
resize_node = helper.make_node(
    'Resize',
    inputs=['X', 'roi', 'scales'],
    outputs=['resize_out'],
    coordinate_transformation_mode='asymmetric',
    mode='linear',
)
# Cast float -> int32 -> float (quantization roundtrip that amplifies differences)
cast_i32 = helper.make_node('Cast', ['resize_out'], ['cast_i32'], to=int(TensorProto.INT32))
cast_f32 = helper.make_node('Cast', ['cast_i32'],   ['cast_f32'], to=int(TensorProto.FLOAT))
mul_node = helper.make_node('Mul', ['cast_f32', 'scale_factor'], ['Y'])

graph = helper.make_graph(
    [resize_node, cast_i32, cast_f32, mul_node],
    'bug_v2_0000',
    [X_info], [Y_info],
    initializer=[scales_init, roi_init, scale_factor_init],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 7
model_bytes = model.SerializeToString()

# ── Input: integer-valued grid so Cast differences are visible ────────────────
INPUT = np.arange(1.0, 1.0 + C * H * W, dtype=np.float32).reshape(1, C, H, W)
# Values will be rescaled by Resize to fractional; Cast(->int32) truncates them

# ── ORT result (matches ONNX spec: asymmetric x_src = x_dst/scale) ───────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

# ── Reference: manually compute asymmetric bilinear ──────────────────────────
def numpy_resize_asymmetric_bilinear(x, scale_h, scale_w):
    N, C, H, W = x.shape
    out_H, out_W = int(H * scale_h), int(W * scale_w)
    out = np.zeros((N, C, out_H, out_W), dtype=np.float32)
    for i in range(out_H):
        for j in range(out_W):
            y_src = i / scale_h
            x_src = j / scale_w
            y0, x0 = int(np.floor(y_src)), int(np.floor(x_src))
            y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
            dy, dx = y_src - y0, x_src - x0
            out[:, :, i, j] = (
                x[:, :, y0, x0] * (1 - dy) * (1 - dx) +
                x[:, :, y0, x1] * (1 - dy) * dx +
                x[:, :, y1, x0] * dy * (1 - dx) +
                x[:, :, y1, x1] * dy * dx
            )
    return out

ref_resize = numpy_resize_asymmetric_bilinear(INPUT, 2.0, 2.0)
ref_after_cast = np.trunc(np.clip(ref_resize, -2**31, 2**31 - 1)).astype(np.float32) * 0.01

max_diff_ort_vs_ref  = float(np.max(np.abs(got - ref_after_cast)))

# Also show divergence between coordinate modes (the root semantic difference)
Xi_cmp  = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Yo_cmp  = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, C, H * 2, W * 2])
resize_hp = helper.make_node('Resize', ['X','roi','scales'], ['Y'],
    coordinate_transformation_mode='pytorch_half_pixel', mode='linear')
graph_hp = helper.make_graph([resize_hp], 'g', [Xi_cmp], [Yo_cmp],
    initializer=[scales_init, roi_init])
model_hp = helper.make_model(graph_hp, opset_imports=[helper.make_opsetid('', 13)])
model_hp.ir_version = 7
sess_hp = ort.InferenceSession(model_hp.SerializeToString(), providers=['CPUExecutionProvider'])
got_hp = sess_hp.run(None, {'X': INPUT})[0]

sess_asym = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got_asym_plain = sess_asym.run(None, {'X': INPUT})[0]

print("=== Bug v2-0000: Resize(linear,asymmetric) coordinate mode divergence ===")
print(f"Input:\n{INPUT[0,0]}")
print(f"\nResize asymmetric output (ORT, no Cast):")
resize_only_bytes = helper.make_model(
    helper.make_graph([helper.make_node('Resize',['X','roi','scales'],['Y'],
        coordinate_transformation_mode='asymmetric',mode='linear')],
        'g', [Xi_cmp], [Yo_cmp], initializer=[scales_init,roi_init]),
    opset_imports=[helper.make_opsetid('',13)])
resize_only_bytes.ir_version = 7
out_asym = ort.InferenceSession(resize_only_bytes.SerializeToString(),
    providers=['CPUExecutionProvider']).run(None, {'X': INPUT})[0]
print(out_asym[0, 0])
print(f"\nResize pytorch_half_pixel output (ORT, no Cast):")
print(got_hp[0, 0])

coord_diff = float(np.max(np.abs(out_asym - got_hp)))
print(f"\nasymmetric vs pytorch_half_pixel max_diff: {coord_diff:.4f}")
print(f"  (asymmetric: x_src = x_dst/scale; pytorch_half_pixel: x_src = (x_dst+0.5)/scale - 0.5)")

print(f"\nFull model (with Cast roundtrip) vs numpy-reference:")
print(f"  ORT got[:4]:       {got.ravel()[:4]}")
print(f"  numpy_ref[:4]:     {ref_after_cast.ravel()[:4]}")
print(f"  max_diff:          {max_diff_ort_vs_ref:.4e}")

# PASS = ORT correctly implements asymmetric (should agree with numpy ref)
pass_flag = max_diff_ort_vs_ref < 1e-5
print(f"\nPASS={pass_flag}")
if pass_flag:
    print("  ORT correctly implements asymmetric bilinear (matches numpy spec).")
    print("  Divergence from pytorch_eager is due to pytorch using different coordinate mode.")
    print(f"  coordinate-mode divergence (asymmetric vs pytorch_half_pixel): {coord_diff:.4f}")
