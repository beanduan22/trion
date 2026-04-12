#!/usr/bin/env python3
"""
Minimal repro for bug v2-0031 (uid=0031).
Compiler  : OnnxRuntime
Patterns  : Resize(nearest_ceil, half_pixel) + transpose-cancel + reshape-roundtrip + 4D-matmul + LogSumExp
Oracle    : ORT_ENABLE_ALL vs ORT_DISABLE_ALL and numpy reference.
Note      : Original bug compares ORT vs pytorch_eager (onnx2torch).
            ORT_ENABLE_ALL == ORT_DISABLE_ALL in both original and minimal repro.
            Divergence from pytorch_eager is likely an onnx2torch Resize coordinate interpretation error.
"""
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto
from onnx import shape_inference
import onnxruntime as ort

np.random.seed(1)

# ── constants ─────────────────────────────────────────────────────────────────
C, H, W = 3, 4, 4
H2, W2 = H * 2, W * 2

mm_w_val    = (np.random.randn(1, 1, W2, W2) * 0.1).astype(np.float32)
lse_ax_val  = np.array([-1], dtype=np.int64)
sq_ax_val   = np.array([-1], dtype=np.int64)
scales_val  = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
roi_val     = np.array([], dtype=np.float32)
flat_val    = np.array([C, H * W], dtype=np.int64)
back_val    = np.array([1, C, H, W], dtype=np.int64)

# ── model ─────────────────────────────────────────────────────────────────────
# Input [1, C, H, W]
# Transpose [0,2,3,1] then [0,3,1,2]  → identity (cancel pair)
# Reshape → flat [C, H*W] → back [1, C, H, W]  (roundtrip)
# Resize nearest_ceil ×2  → [1, C, H2, W2]
# MatMul 4D [1,1,W2,W2]   → [1, C, H2, W2]
# LogSumExp over last axis → [1, C, H2]
X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [1, C, H2])

nodes = [
    oh.make_node('Transpose', ['X'],    ['tr1'], perm=[0, 2, 3, 1]),
    oh.make_node('Transpose', ['tr1'],  ['tr2'], perm=[0, 3, 1, 2]),
    oh.make_node('Reshape',   ['tr2', 'flat'], ['r1']),
    oh.make_node('Reshape',   ['r1',  'back'], ['r2']),
    oh.make_node('Resize', ['r2', 'roi', 'scales'], ['rsz'],
                 coordinate_transformation_mode='half_pixel',
                 mode='nearest',
                 nearest_mode='ceil'),
    oh.make_node('MatMul', ['rsz', 'mm_w'], ['mm']),
    # LogSumExp: keepdims=1 for mx so Sub can broadcast, squeeze at end
    oh.make_node('ReduceMax', ['mm'],     ['mx_k1'], axes=[-1], keepdims=1),
    oh.make_node('Sub',       ['mm', 'mx_k1'], ['sub']),
    oh.make_node('Exp',       ['sub'],    ['exp']),
    oh.make_node('ReduceSum', ['exp', 'lse_ax'], ['rsum'], keepdims=0),
    oh.make_node('Log',       ['rsum'],   ['logsum']),
    oh.make_node('Squeeze',   ['mx_k1', 'sq_ax'], ['mx_sq']),
    oh.make_node('Add',       ['logsum', 'mx_sq'], ['Y']),
]

inits = [
    nph.from_array(mm_w_val,   name='mm_w'),
    nph.from_array(lse_ax_val, name='lse_ax'),
    nph.from_array(sq_ax_val,  name='sq_ax'),
    nph.from_array(scales_val, name='scales'),
    nph.from_array(roi_val,    name='roi'),
    nph.from_array(flat_val,   name='flat'),
    nph.from_array(back_val,   name='back'),
]

graph = oh.make_graph(nodes, 'bug_v2_0031', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.random.randn(1, C, H, W).astype(np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
def nearest_ceil_resize_1d(arr, scale):
    """half_pixel + nearest_ceil in 1D (last axis)."""
    n_out = int(arr.shape[-1] * scale)
    n_in  = arr.shape[-1]
    out   = np.empty(arr.shape[:-1] + (n_out,), dtype=arr.dtype)
    for i in range(n_out):
        x_orig = (i + 0.5) / scale - 0.5
        src    = int(np.ceil(x_orig))
        src    = max(0, min(n_in - 1, src))
        out[..., i] = arr[..., src]
    return out

def nearest_ceil_resize_2d(arr, scale_h, scale_w):
    r = nearest_ceil_resize_1d(arr,                         scale_h)  # H axis (dim 2)
    r = nearest_ceil_resize_1d(r.transpose(0,1,3,2), scale_w)        # W axis
    return r.transpose(0, 1, 3, 2)

tr = x  # transpose cancel
rsz_np = nearest_ceil_resize_2d(tr, 2.0, 2.0)           # [1, C, H2, W2]
mm_np  = rsz_np @ mm_w_val[0, 0]                         # [1, C, H2, W2]
mx_np  = mm_np.max(axis=-1, keepdims=True)
lse_np = np.log(np.sum(np.exp(mm_np - mx_np), axis=-1)) + mx_np[..., 0]

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
all_vs_numpy = np.allclose(o_all, lse_np, atol=tol)

print(f"ORT_ENABLE_ALL  sample [0,0,0]: {o_all[0,0,0]:.6f}")
print(f"ORT_DISABLE_ALL sample [0,0,0]: {o_dis[0,0,0]:.6f}")
print(f"numpy reference sample [0,0,0]: {lse_np[0,0,0]:.6f}")
print(f"ORT_ALL == ORT_DIS  : {all_vs_dis}")
print(f"ORT_ALL == numpy    : {all_vs_numpy}")
PASS = all_vs_numpy
print(f"PASS={PASS}")
