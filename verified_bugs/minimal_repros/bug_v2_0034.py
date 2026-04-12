#!/usr/bin/env python3
"""
Minimal repro for bug v2-0034 (uid=0034).
Compiler  : OnnxRuntime
Patterns  : LayerNorm + temperature-div + self-add chain + self-sub-zero + TopK(k=1, axis=-1)
            + Tile + Reshape + InstanceNorm
Bug desc  : SUSPECT — self-sub-zero (x-x→0) elimination + temperature layernorm.
            Original divergence is from pytorch_eager (onnx2torch), ORT_ENABLE_ALL == ORT_DISABLE_ALL.
Oracle    : ORT_ENABLE_ALL vs ORT_DISABLE_ALL and numpy reference.
"""
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto
from onnx import shape_inference
import onnxruntime as ort

np.random.seed(4)

# ── constants ─────────────────────────────────────────────────────────────────
B, D = 1, 8

ln_sc_val  = np.ones(D, dtype=np.float32)
ln_b_val   = np.zeros(D, dtype=np.float32)
temp_val   = np.array([2.25], dtype=np.float32)
bias_val   = np.zeros((1, D), dtype=np.float32)
tk_k_val   = np.array([1], dtype=np.int64)
tk_rep_val = np.array([1, D], dtype=np.int64)
in3d_val   = np.array([1, 1, D], dtype=np.int64)
in2d_val   = np.array([1, D], dtype=np.int64)
in_sc_val  = np.ones(1, dtype=np.float32)
in_b_val   = np.zeros(1, dtype=np.float32)

# ── model ─────────────────────────────────────────────────────────────────────
# LayerNorm → Div(temp) → Add(bias)
# → self-add ×3 (×4 accumulation)
# → Sub(self) → Add(original)   — the self-sub-zero pattern
# → TopK(k=1, axis=-1, smallest) → Tile → Reshape [1,1,D]
# → InstanceNorm → Reshape [1,D]
X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, D])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, D])

nodes = [
    oh.make_node('LayerNormalization', ['X', 'ln_sc', 'ln_b'], ['ln_out'],
                 axis=-1, epsilon=1e-6),
    oh.make_node('Div',  ['ln_out', 'temp'],     ['div_out']),
    oh.make_node('Add',  ['div_out', 'bias'],    ['lnt_out']),
    oh.make_node('Add',  ['lnt_out', 'lnt_out'], ['a1']),
    oh.make_node('Add',  ['a1',      'lnt_out'], ['a2']),
    oh.make_node('Add',  ['a2',      'lnt_out'], ['a3']),
    oh.make_node('Sub',  ['a3',      'a3'],      ['sub_zero']),
    oh.make_node('Add',  ['a3',      'sub_zero'], ['sub_out']),
    oh.make_node('TopK', ['sub_out', 'tk_k'],    ['tk_vals', 'tk_idx'],
                 axis=-1, largest=0, sorted=0),
    oh.make_node('Tile', ['tk_vals', 'tk_rep'],  ['tiled']),
    oh.make_node('Reshape', ['tiled', 'in3d'],   ['in3d_t']),
    oh.make_node('InstanceNormalization', ['in3d_t', 'in_sc', 'in_b'], ['in_out'],
                 epsilon=1e-5),
    oh.make_node('Reshape', ['in_out', 'in2d'],  ['Y']),
]

inits = [
    nph.from_array(ln_sc_val,  name='ln_sc'),
    nph.from_array(ln_b_val,   name='ln_b'),
    nph.from_array(temp_val,   name='temp'),
    nph.from_array(bias_val,   name='bias'),
    nph.from_array(tk_k_val,   name='tk_k'),
    nph.from_array(tk_rep_val, name='tk_rep'),
    nph.from_array(in3d_val,   name='in3d'),
    nph.from_array(in2d_val,   name='in2d'),
    nph.from_array(in_sc_val,  name='in_sc'),
    nph.from_array(in_b_val,   name='in_b'),
]

graph = oh.make_graph(nodes, 'bug_v2_0034', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.array([[0.5, -1.0, 2.0, -0.5, 0.1, 1.5, -2.0, 0.3]], dtype=np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
mu   = x.mean(axis=-1, keepdims=True)
std  = np.sqrt(x.var(axis=-1, keepdims=True) + 1e-6)
ln   = (x - mu) / std * ln_sc_val + ln_b_val
lnt  = ln / float(temp_val[0]) + bias_val
a3   = lnt * 4.0    # a1=2×lnt, a2=3×, a3=4×
sub  = np.zeros_like(a3)   # x - x = 0
out  = a3 + sub             # a3 (unchanged)

# TopK k=1 axis=-1 smallest → [B, 1]
min_v = out.min(axis=-1, keepdims=True)       # [1, 1]
tiled = np.tile(min_v, [1, D])                # [1, D] (broadcast single value)
in3d  = tiled.reshape(1, 1, D)

# InstanceNorm [1, C=1, spatial=D]: all same values → mean = value, std = 0 → result = 0
mu_in  = in3d.mean(axis=-1, keepdims=True)
std_in = np.sqrt(in3d.var(axis=-1, keepdims=True) + 1e-5)
norm   = (in3d - mu_in) / std_in             # [1, 1, D]
out_np = norm.reshape(1, D).astype(np.float32)

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
# SUSPECT: pytorch_eager may compute InstanceNorm differently causing the original divergence
PASS = all_vs_numpy
print(f"PASS={PASS}")
