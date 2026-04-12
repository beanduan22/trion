#!/usr/bin/env python3
"""
Minimal repro for bug v2-0038 (uid=0038).
Compiler  : OnnxRuntime
Patterns  : residual-relu + mul_by_one chain + sub_self (x-x→0) + gather-layernorm
Bug desc  : sub_self (x-x→0) elimination + mul_by_one (x*1→x) elimination pattern.
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

np.random.seed(8)

# ── constants ─────────────────────────────────────────────────────────────────
B, D = 1, 8

one_val    = np.array([1.0], dtype=np.float32)
tk_k_val   = np.array([1], dtype=np.int64)
tk_rep_val = np.array([1, D], dtype=np.int64)
embed_val  = (np.random.randn(4, D) * 0.01).astype(np.float32)
idx_val    = np.array([0], dtype=np.int64)
ln_g_val   = np.ones(D, dtype=np.float32)
ln_b_val   = np.zeros(D, dtype=np.float32)

# ── model ─────────────────────────────────────────────────────────────────────
# Relu(x) → Add(x, relu) → Relu
# mul × 1 × 1 × 1  (mul_by_one chain)
# Sub(self) → Add(val, zero)   (self-sub zero)
# TopK(k=1, axis=-1) → Tile → Gather(embed[0]) → Add → LayerNorm
X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, D])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, D])

nodes = [
    oh.make_node('Relu',   ['X'],           ['r1']),
    oh.make_node('Add',    ['r1', 'X'],     ['ad']),
    oh.make_node('Relu',   ['ad'],          ['r2']),
    oh.make_node('Mul',    ['r2',  'one'],  ['m1']),
    oh.make_node('Mul',    ['m1',  'one'],  ['m2']),
    oh.make_node('Mul',    ['m2',  'one'],  ['m3']),
    oh.make_node('Sub',    ['m3',  'm3'],   ['sub_z']),
    oh.make_node('Add',    ['m3',  'sub_z'],['sub_out']),
    oh.make_node('TopK',   ['sub_out', 'tk_k'], ['tk_vals', 'tk_idx'],
                 axis=-1, largest=0, sorted=0),
    oh.make_node('Tile',   ['tk_vals', 'tk_rep'], ['tiled']),
    oh.make_node('Gather', ['embed', 'idx'], ['gth'], axis=0),
    oh.make_node('Add',    ['gth', 'tiled'], ['added']),
    oh.make_node('LayerNormalization', ['added', 'ln_g', 'ln_b'], ['Y'],
                 axis=-1, epsilon=1e-6),
]

inits = [
    nph.from_array(one_val,    name='one'),
    nph.from_array(tk_k_val,   name='tk_k'),
    nph.from_array(tk_rep_val, name='tk_rep'),
    nph.from_array(embed_val,  name='embed'),
    nph.from_array(idx_val,    name='idx'),
    nph.from_array(ln_g_val,   name='ln_g'),
    nph.from_array(ln_b_val,   name='ln_b'),
]

graph = oh.make_graph(nodes, 'bug_v2_0038', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.array([[0.5, -1.0, 2.0, -0.5, 0.1, 1.5, -2.0, 0.3]], dtype=np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
r1      = np.maximum(x, 0)
ad      = r1 + x
r2      = np.maximum(ad, 0)
m3      = r2 * 1.0 * 1.0 * 1.0   # mul_by_one identity
sub_z   = m3 - m3                  # zero
sub_out = m3 + sub_z               # = m3

# TopK k=1 axis=-1 smallest → [B, 1]
min_v   = sub_out.min(axis=-1, keepdims=True)
tiled   = np.tile(min_v, [1, D])   # [B, D]

# Gather embed[0]
gth     = embed_val[0:1]           # [1, D]
added   = gth + tiled

mu      = added.mean(axis=-1, keepdims=True)
std     = np.sqrt(added.var(axis=-1, keepdims=True) + 1e-6)
out_np  = ((added - mu) / std * ln_g_val + ln_b_val).astype(np.float32)

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

print(f"ORT_ENABLE_ALL  flat: {o_all.ravel()}")
print(f"ORT_DISABLE_ALL flat: {o_dis.ravel()}")
print(f"numpy reference flat: {out_np.ravel()}")
print(f"ORT_ALL == ORT_DIS  : {all_vs_dis}")
print(f"ORT_ALL == numpy    : {all_vs_numpy}")
PASS = all_vs_numpy
print(f"PASS={PASS}")
