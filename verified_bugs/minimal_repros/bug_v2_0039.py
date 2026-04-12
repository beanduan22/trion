#!/usr/bin/env python3
"""
Minimal repro for bug v2-0039 (uid=0039).
Compiler  : OnnxRuntime
Patterns  : 4D-matmul + ReduceSum(middle, keepdims) + channel-concat + linear-Resize
            + nearest_ceil-Resize ×2
Bug desc  : Resize(nearest_ceil) + ReduceSum(middle) + pixel propagation.
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

np.random.seed(9)

# ── constants ─────────────────────────────────────────────────────────────────
# Use small spatial dims to keep model fast
B, C, H, W = 1, 2, 4, 4

mm1_w_val  = (np.random.randn(1, 1, W, W) * 0.1).astype(np.float32)
rsm_ax_val = np.array([2], dtype=np.int64)          # ReduceSum axis=2 (H dim)
shift_val  = np.full((1, C, 1, 1), 0.1, dtype=np.float32)  # [1, C, 1, 1]
linear_scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # no-op resize
linear_roi    = np.array([], dtype=np.float32)
mm2_w_val  = mm1_w_val.copy()
ceil_scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)   # ×2 spatial
ceil_roi    = np.array([], dtype=np.float32)

# ── model ─────────────────────────────────────────────────────────────────────
# Input [1, C, H, W]
# MatMul 4D → [1, C, H, W]
# ReduceSum axis=2 keepdims=1 → [1, C, 1, W]
# Add [1,C,H,W] + [1,C,1,W] → [1,C,H,W]   (broadcast)
# Add + shift[1,C,1,1] → shifted [1,C,H,W]
# Concat([rsm_out, shifted], axis=1) → [1, 2C, H, W]
# Resize linear scales=1 → [1, 2C, H, W]   (no-op, identity)
# Concat([rsz, cat1], axis=1) → [1, 4C, H, W]
# MatMul 4D → [1, 4C, H, W]
# Resize nearest_ceil ×2 → [1, 4C, 2H, 2W]
C2 = C * 2
C4 = C * 4
H2, W2 = H * 2, W * 2

X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, C, H, W])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, C4, H2, W2])

nodes = [
    oh.make_node('MatMul',    ['X', 'mm1_w'],          ['mm1']),
    oh.make_node('ReduceSum', ['mm1', 'rsm_ax'],        ['rsm_red'],  keepdims=1),
    oh.make_node('Add',       ['mm1', 'rsm_red'],       ['rsm_out']),
    oh.make_node('Add',       ['rsm_out', 'shift'],     ['shifted']),
    oh.make_node('Concat',    ['rsm_out', 'shifted'],   ['cat1'],     axis=1),
    oh.make_node('Resize',    ['cat1', 'lin_roi', 'lin_scales'], ['rsz'],
                 coordinate_transformation_mode='half_pixel',
                 mode='linear'),
    oh.make_node('Concat',    ['rsz', 'cat1'],          ['cat2'],     axis=1),
    oh.make_node('MatMul',    ['cat2', 'mm2_w'],        ['mm2']),
    oh.make_node('Resize',    ['mm2', 'ceil_roi', 'ceil_scales'], ['Y'],
                 coordinate_transformation_mode='half_pixel',
                 mode='nearest',
                 nearest_mode='ceil'),
]

inits = [
    nph.from_array(mm1_w_val,      name='mm1_w'),
    nph.from_array(rsm_ax_val,     name='rsm_ax'),
    nph.from_array(shift_val,      name='shift'),
    nph.from_array(linear_scales,  name='lin_scales'),
    nph.from_array(linear_roi,     name='lin_roi'),
    nph.from_array(mm2_w_val,      name='mm2_w'),
    nph.from_array(ceil_scales,    name='ceil_scales'),
    nph.from_array(ceil_roi,       name='ceil_roi'),
]

graph = oh.make_graph(nodes, 'bug_v2_0039', [X], [Y], inits)
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

mm1       = x @ mm1_w_val[0, 0]                       # [1, C, H, W]
rsm_red   = mm1.sum(axis=2, keepdims=True)             # [1, C, 1, W]
rsm_out   = mm1 + rsm_red                              # [1, C, H, W]
shifted   = rsm_out + shift_val                        # [1, C, H, W]
cat1      = np.concatenate([rsm_out, shifted], axis=1) # [1, 2C, H, W]
# Linear resize scales=1 → identity
rsz       = cat1.copy()                                # [1, 2C, H, W]
cat2      = np.concatenate([rsz, cat1], axis=1)        # [1, 4C, H, W]
mm2       = cat2 @ mm2_w_val[0, 0]                    # [1, 4C, H, W]
out_np    = nearest_ceil_2d(mm2, 2.0, 2.0).astype(np.float32)  # [1, 4C, 2H, 2W]

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
