#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0036
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : log1p(|x|) * x + matmul+bias+GELU + Max(multi-const) + double-LayerNorm + L2-norm
Root cause : n/a
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto
from onnx import shape_inference
import onnxruntime as ort
from scipy.special import erf as sp_erf

np.random.seed(6)

# ── constants ─────────────────────────────────────────────────────────────────
B, S, D = 1, 4, 8   # batch, seq, dim

one_val     = np.array([1.0], dtype=np.float32)
W_val       = (np.random.randn(D, D) * 0.1).astype(np.float32)
b_val       = np.zeros((1, 1, D), dtype=np.float32)
sqrt2_val   = np.array([1.41421356], dtype=np.float32)
half_val    = np.array([0.5], dtype=np.float32)
erf_one_val = np.array([1.0], dtype=np.float32)
c0_val      = np.full((1, S, D), 0.1, dtype=np.float32)
c1_val      = np.full((1, S, D), 0.2, dtype=np.float32)
c2_val      = np.full((1, S, D), 0.3, dtype=np.float32)
ln1_g_val   = np.ones(D, dtype=np.float32)
ln1_b_val   = np.zeros(D, dtype=np.float32)
scale_val   = np.full((1, S, D), 1.1, dtype=np.float32)
ln2_g_val   = np.ones(D, dtype=np.float32)
ln2_b_val   = np.zeros(D, dtype=np.float32)
l2_eps_val  = np.array([1e-6], dtype=np.float32)
l2_sc_val   = np.array([[[1.0]]], dtype=np.float32)

# ── model ─────────────────────────────────────────────────────────────────────
# Abs(x) + 1 → Log → Mul(x) = log1p(|x|) * x
# MatMul → Add bias → GELU
# Max(out, c0, c1, c2)
# LayerNorm → Mul(scale) → LayerNorm
# L2-norm (ReduceL2 → Add eps → Div → Mul scale)
X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, S, D])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, S, D])

nodes = [
    oh.make_node('Abs',  ['X'],            ['abs_x']),
    oh.make_node('Add',  ['abs_x', 'one'], ['abs1']),
    oh.make_node('Log',  ['abs1'],         ['log_out']),
    oh.make_node('Mul',  ['log_out', 'X'], ['l1p_out']),
    oh.make_node('MatMul', ['l1p_out', 'W'], ['mm']),
    oh.make_node('Add',    ['mm', 'b'],      ['mm_b']),
    oh.make_node('Div',    ['mm_b', 'sqrt2'], ['div']),
    oh.make_node('Erf',    ['div'],           ['erf']),
    oh.make_node('Add',    ['erf', 'erf_one'], ['ep1']),
    oh.make_node('Mul',    ['mm_b', 'half'],   ['hf']),
    oh.make_node('Mul',    ['hf', 'ep1'],      ['gelu_out']),
    oh.make_node('Max',    ['gelu_out', 'c0', 'c1', 'c2'], ['max_out']),
    oh.make_node('LayerNormalization', ['max_out', 'ln1_g', 'ln1_b'], ['ln1_out'],
                 axis=-1, epsilon=1e-6),
    oh.make_node('Mul',    ['ln1_out', 'scale'], ['sc_out']),
    oh.make_node('LayerNormalization', ['sc_out', 'ln2_g', 'ln2_b'], ['ln2_out'],
                 axis=-1, epsilon=1e-6),
    oh.make_node('ReduceL2', ['ln2_out'], ['l2_norm'], axes=[-1], keepdims=1),
    oh.make_node('Add',      ['l2_norm', 'l2_eps'], ['l2_eps_out']),
    oh.make_node('Div',      ['ln2_out', 'l2_eps_out'], ['l2_div']),
    oh.make_node('Mul',      ['l2_div', 'l2_sc'], ['Y']),
]

inits = [
    nph.from_array(one_val,     name='one'),
    nph.from_array(W_val,       name='W'),
    nph.from_array(b_val,       name='b'),
    nph.from_array(sqrt2_val,   name='sqrt2'),
    nph.from_array(half_val,    name='half'),
    nph.from_array(erf_one_val, name='erf_one'),
    nph.from_array(c0_val,      name='c0'),
    nph.from_array(c1_val,      name='c1'),
    nph.from_array(c2_val,      name='c2'),
    nph.from_array(ln1_g_val,   name='ln1_g'),
    nph.from_array(ln1_b_val,   name='ln1_b'),
    nph.from_array(scale_val,   name='scale'),
    nph.from_array(ln2_g_val,   name='ln2_g'),
    nph.from_array(ln2_b_val,   name='ln2_b'),
    nph.from_array(l2_eps_val,  name='l2_eps'),
    nph.from_array(l2_sc_val,   name='l2_sc'),
]

graph = oh.make_graph(nodes, 'bug_v2_0036', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.random.randn(B, S, D).astype(np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
l1p      = np.log(np.abs(x) + 1.0) * x
mm_b     = l1p @ W_val + b_val
gelu     = mm_b * 0.5 * (1.0 + sp_erf(mm_b / float(sqrt2_val[0])))
max_out  = np.maximum(gelu, np.maximum(c0_val, np.maximum(c1_val, c2_val)))

mu1  = max_out.mean(axis=-1, keepdims=True)
std1 = np.sqrt(max_out.var(axis=-1, keepdims=True) + 1e-6)
ln1  = (max_out - mu1) / std1 * ln1_g_val + ln1_b_val

sc   = ln1 * scale_val
mu2  = sc.mean(axis=-1, keepdims=True)
std2 = np.sqrt(sc.var(axis=-1, keepdims=True) + 1e-6)
ln2  = (sc - mu2) / std2 * ln2_g_val + ln2_b_val

l2_norm = np.sqrt((ln2 ** 2).sum(axis=-1, keepdims=True))  # [B, S, 1]
l2_eps_out = l2_norm + 1e-6                                 # [B, S, 1]
out_np   = (ln2 / l2_eps_out * l2_sc_val).astype(np.float32)

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

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
