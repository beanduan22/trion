#!/usr/bin/env python3
"""
Minimal repro for bug v2-0037 (uid=0037).
Compiler  : OnnxRuntime
Patterns  : Resize(nearest_ceil, half_pixel) + back-to-back 4D matmul + Where(mask) + Pad(edge)
Bug desc  : Resize(nearest_ceil) + back-to-back 4D matmul pattern.
            Original divergence is ORT vs pytorch_eager (onnx2torch). ORT_ENABLE_ALL == ORT_DISABLE_ALL.
Oracle    : ORT_ENABLE_ALL vs ORT_DISABLE_ALL and numpy reference.
"""
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto
from onnx import shape_inference
import onnxruntime as ort

np.random.seed(7)

# ── constants ─────────────────────────────────────────────────────────────────
B, C, H, W = 1, 2, 4, 4
H2, W2 = H * 2, W * 2  # after resize

mm1_w_val = (np.random.randn(1, 1, W2, W2) * 0.1).astype(np.float32)
mm2_w_val = mm1_w_val.copy()
zero_val  = np.array([0.0], dtype=np.float32)
fill_val  = np.array([-1.0], dtype=np.float32)
# Pad: edge mode, 1 pixel on each spatial edge
pads_val  = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
scales_val = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
roi_val    = np.array([], dtype=np.float32)

# ── model ─────────────────────────────────────────────────────────────────────
# Resize nearest_ceil ×2 → MatMul 4D → MatMul 4D
# → Greater(0) → Where(mask, val, -1) → Mul(val) → Pad(edge, 1)
X  = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, C, H, W])
Y  = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, C, H2 + 2, W2 + 2])

nodes = [
    oh.make_node('Resize', ['X', 'roi', 'scales'], ['rsz'],
                 coordinate_transformation_mode='half_pixel',
                 mode='nearest',
                 nearest_mode='ceil'),
    oh.make_node('MatMul', ['rsz',  'mm1_w'], ['mm1']),
    oh.make_node('MatMul', ['mm1',  'mm2_w'], ['mm2']),
    oh.make_node('Greater',['mm2',  'zero'],  ['gt']),
    oh.make_node('Where',  ['gt', 'mm2', 'fill'], ['wh']),
    oh.make_node('Mul',    ['wh',  'mm2'],    ['mul_out']),
    oh.make_node('Pad',    ['mul_out', 'pads'], ['Y'], mode='edge'),
]

inits = [
    nph.from_array(mm1_w_val, name='mm1_w'),
    nph.from_array(mm2_w_val, name='mm2_w'),
    nph.from_array(zero_val,  name='zero'),
    nph.from_array(fill_val,  name='fill'),
    nph.from_array(pads_val,  name='pads'),
    nph.from_array(scales_val, name='scales'),
    nph.from_array(roi_val,   name='roi'),
]

graph = oh.make_graph(nodes, 'bug_v2_0037', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.random.randn(B, C, H, W).astype(np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
def nearest_ceil_1d(arr, scale):
    n_out, n_in = int(arr.shape[-1] * scale), arr.shape[-1]
    out = np.empty(arr.shape[:-1] + (n_out,), dtype=arr.dtype)
    for i in range(n_out):
        src = int(np.ceil((i + 0.5) / scale - 0.5))
        out[..., i] = arr[..., max(0, min(n_in - 1, src))]
    return out

def nearest_ceil_2d(a, sh, sw):
    r = nearest_ceil_1d(a, sh)
    r = nearest_ceil_1d(r.transpose(0,1,3,2), sw).transpose(0,1,3,2)
    return r

rsz     = nearest_ceil_2d(x, 2.0, 2.0)
mm1     = rsz  @ mm1_w_val[0, 0]
mm2     = mm1  @ mm2_w_val[0, 0]
gt      = mm2 > 0.0
wh      = np.where(gt, mm2, fill_val)
mul_out = wh * mm2
# Edge pad: 1 pixel on each spatial side
pad_out = np.pad(mul_out, ((0,0),(0,0),(1,1),(1,1)), mode='edge')
out_np  = pad_out.astype(np.float32)

# ── ORT runs ──────────────────────────────────────────────────────────────────
def ort_run(opt_level):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                providers=['CPUExecutionProvider'])
    return sess.run(None, {'X': x})[0]

o_all = ort_run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
o_dis = ort_run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

tol = 1e-4
all_vs_dis   = np.allclose(o_all, o_dis, atol=tol)
all_vs_numpy = np.allclose(o_all, out_np, atol=tol)

print(f"ORT_ENABLE_ALL  sample flat[0]: {o_all.ravel()[0]:.6f}")
print(f"ORT_DISABLE_ALL sample flat[0]: {o_dis.ravel()[0]:.6f}")
print(f"numpy reference sample flat[0]: {out_np.ravel()[0]:.6f}")
print(f"ORT_ALL == ORT_DIS  : {all_vs_dis}")
print(f"ORT_ALL == numpy    : {all_vs_numpy}")
PASS = all_vs_numpy
print(f"PASS={PASS}")
