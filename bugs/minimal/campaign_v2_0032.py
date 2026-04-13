#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0032
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : mul_by_zero skip + double-shift + LayerNorm + matmul+bias + dropout + squeeze/unsqueeze + pow_by_one
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

np.random.seed(2)

# ── constants ─────────────────────────────────────────────────────────────────
B, S, D = 1, 4, 8  # batch, seq, dim

zeros_val  = np.zeros((1, S, D), dtype=np.float32)
scale_val  = np.full((1, S, D), 1.05, dtype=np.float32)
shift1_val = np.full((1, 1, D), 0.01, dtype=np.float32)
shift2_val = np.full((1, 1, D), 0.02, dtype=np.float32)
ln_g_val   = np.ones(D, dtype=np.float32)
ln_b_val   = np.zeros(D, dtype=np.float32)
W_val      = (np.random.randn(D, D) * 0.1).astype(np.float32)
b_fc_val   = np.zeros(D, dtype=np.float32)
ratio_val  = np.array(0.0, dtype=np.float32)  # dropout ratio=0 → identity
shift3_val = np.full((1, 1, D), 0.01, dtype=np.float32)
ax1_val    = np.array([3], dtype=np.int64)
ax2_val    = np.array([3], dtype=np.int64)
one_val    = np.array([1], dtype=np.int64)

# ── model ─────────────────────────────────────────────────────────────────────
# x * 0  → x_zero  (multiply-by-zero)
# x_zero + scale → add_const
# add_const + shift1 → a1
# a1 + shift2 → a2
# LayerNorm(a2, g, b) → ln_out
# MatMul → Add bias → Dropout(0) → Add shift → Unsqueeze/Squeeze roundtrip
# Pow(·, 1) + · → out
X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, S, D])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, S, D])

nodes = [
    oh.make_node('Mul',  ['X', 'zeros'],  ['x_zero']),
    oh.make_node('Add',  ['x_zero', 'scale'], ['add_sc']),
    oh.make_node('Add',  ['add_sc', 'shift1'], ['a1']),
    oh.make_node('Add',  ['a1', 'shift2'],  ['a2']),
    oh.make_node('LayerNormalization', ['a2', 'ln_g', 'ln_b'], ['ln_out'],
                 axis=-1, epsilon=1e-6),
    oh.make_node('MatMul', ['ln_out', 'W'],    ['mm']),
    oh.make_node('Add',    ['mm', 'b_fc'],     ['mm_b']),
    oh.make_node('Dropout', ['mm_b', 'ratio'], ['dr']),
    oh.make_node('Add',    ['dr', 'shift3'],   ['shifted']),
    oh.make_node('Unsqueeze', ['shifted', 'ax1'], ['unsq']),
    oh.make_node('Squeeze',   ['unsq',    'ax2'], ['sq_out']),
    oh.make_node('Pow',       ['sq_out', 'one'],  ['pw']),
    oh.make_node('Add',       ['pw', 'sq_out'],   ['Y']),
]

inits = [
    nph.from_array(zeros_val,  name='zeros'),
    nph.from_array(scale_val,  name='scale'),
    nph.from_array(shift1_val, name='shift1'),
    nph.from_array(shift2_val, name='shift2'),
    nph.from_array(ln_g_val,   name='ln_g'),
    nph.from_array(ln_b_val,   name='ln_b'),
    nph.from_array(W_val,      name='W'),
    nph.from_array(b_fc_val,   name='b_fc'),
    nph.from_array(ratio_val,  name='ratio'),
    nph.from_array(shift3_val, name='shift3'),
    nph.from_array(ax1_val,    name='ax1'),
    nph.from_array(ax2_val,    name='ax2'),
    nph.from_array(one_val,    name='one'),
]

graph = oh.make_graph(nodes, 'bug_v2_0032', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.random.randn(B, S, D).astype(np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
x_zero  = x * zeros_val
add_sc  = x_zero + scale_val
a1      = add_sc + shift1_val
a2      = a1 + shift2_val

mu  = a2.mean(axis=-1, keepdims=True)
std = np.sqrt(a2.var(axis=-1, keepdims=True) + 1e-6)
ln  = (a2 - mu) / std * ln_g_val + ln_b_val

mm_b    = ln @ W_val + b_fc_val   # dropout(0) = identity
shifted = mm_b + shift3_val
sq_out  = shifted                  # unsqueeze/squeeze roundtrip = identity
pw      = sq_out ** 1
out_np  = pw + sq_out

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
