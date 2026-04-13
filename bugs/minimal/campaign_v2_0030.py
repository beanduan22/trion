#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0030
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : reciprocal_mul, cumsum_last_axis, gather_layernorm, matmul_bias_gelu
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

np.random.seed(0)

# ── build model ──────────────────────────────────────────────────────────────
# Input: [1, 8] float32
# Pipeline:
#   x * (1/denom)  → mul_by_rc  → cumsum(axis=1)  → +embed  → layernorm → matmul → gelu

N = 8
H = 4  # hidden after matmul

denom_val   = np.full((1, N), 1.1, dtype=np.float32)
rc_val      = np.array([0.198], dtype=np.float32)
bias_val    = np.zeros((1, N), dtype=np.float32)
embed_val   = (np.random.randn(1, N) * 0.01).astype(np.float32)
ln_g_val    = np.ones(N, dtype=np.float32)
ln_b_val    = np.zeros(N, dtype=np.float32)
W_val       = (np.random.randn(N, H) * 0.1).astype(np.float32)
b_out_val   = np.zeros(H, dtype=np.float32)
sqrt2_val   = np.array([1.41421356], dtype=np.float32)
half_val    = np.array([0.5], dtype=np.float32)
one_val     = np.array([1.0], dtype=np.float32)
cs_ax_val   = np.array(1, dtype=np.int64)

X    = oh.make_tensor_value_info('X', TensorProto.FLOAT, [1, N])
Y    = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [1, H])

nodes = [
    oh.make_node('Reciprocal', ['denom'],    ['rec']),
    oh.make_node('Mul',        ['X', 'rec'], ['m1']),
    oh.make_node('Mul',        ['m1', 'rc'], ['m2']),
    oh.make_node('Add',        ['m2', 'bias'], ['pre_cs']),
    oh.make_node('CumSum',     ['pre_cs', 'cs_ax'], ['cs_out']),
    oh.make_node('Add',        ['cs_out', 'embed'], ['ln_in']),
    oh.make_node('LayerNormalization', ['ln_in', 'ln_g', 'ln_b'], ['ln_out'],
                 axis=-1, epsilon=1e-6),
    oh.make_node('MatMul', ['ln_out', 'W'], ['mm']),
    oh.make_node('Add',    ['mm', 'b_out'], ['mm_b']),
    oh.make_node('Div',    ['mm_b', 'sqrt2'], ['div']),
    oh.make_node('Erf',    ['div'], ['erf']),
    oh.make_node('Add',    ['erf', 'one'], ['ep1']),
    oh.make_node('Mul',    ['mm_b', 'half'], ['hf']),
    oh.make_node('Mul',    ['hf', 'ep1'], ['Y']),
]

inits = [
    nph.from_array(denom_val,  name='denom'),
    nph.from_array(rc_val,     name='rc'),
    nph.from_array(bias_val,   name='bias'),
    nph.from_array(cs_ax_val,  name='cs_ax'),
    nph.from_array(embed_val,  name='embed'),
    nph.from_array(ln_g_val,   name='ln_g'),
    nph.from_array(ln_b_val,   name='ln_b'),
    nph.from_array(W_val,      name='W'),
    nph.from_array(b_out_val,  name='b_out'),
    nph.from_array(sqrt2_val,  name='sqrt2'),
    nph.from_array(half_val,   name='half'),
    nph.from_array(one_val,    name='one'),
]

graph = oh.make_graph(nodes, 'bug_v2_0030', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.array([[0.29, 0.60, 0.15, 0.79, 0.11, -0.53, -0.12, -0.24]], dtype=np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
from scipy.special import erf as sp_erf  # noqa: E402

rec   = 1.0 / denom_val
m1    = x * rec
m2    = m1 * rc_val
pre   = m2 + bias_val
cs    = np.cumsum(pre, axis=1)
ln_in = cs + embed_val

mu  = ln_in.mean(axis=-1, keepdims=True)
std = np.sqrt(ln_in.var(axis=-1, keepdims=True) + 1e-6)
ln  = (ln_in - mu) / std * ln_g_val + ln_b_val

mm_b = ln @ W_val + b_out_val
gelu = mm_b * 0.5 * (1.0 + sp_erf(mm_b / float(sqrt2_val[0])))

# ── ORT runs ──────────────────────────────────────────────────────────────────
def ort_run(opt_level):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                providers=['CPUExecutionProvider'])
    return sess.run(None, {'X': x})[0]

o_all = ort_run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
o_dis = ort_run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

# ── verdict ───────────────────────────────────────────────────────────────────
tol = 1e-4
all_vs_dis  = np.allclose(o_all, o_dis, atol=tol)
all_vs_numpy = np.allclose(o_all, gelu, atol=tol)

print(f"ORT_ENABLE_ALL  : {o_all.ravel()}")
print(f"ORT_DISABLE_ALL : {o_dis.ravel()}")
print(f"numpy reference : {gelu.ravel()}")
print(f"ORT_ALL == ORT_DIS  : {all_vs_dis}")
print(f"ORT_ALL == numpy    : {all_vs_numpy}")
# PASS=True when ORT matches numpy (no ORT correctness bug in this minimal repro)
PASS = all_vs_numpy
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
