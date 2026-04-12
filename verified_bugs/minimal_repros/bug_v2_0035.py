#!/usr/bin/env python3
"""
Minimal repro for bug v2-0035 (uid=0035).
Compiler  : OnnxRuntime
Patterns  : residual-add-relu + erf-chain (tanh→erf→mul) + neg-abs-identity + 4D-matmul + CumSum(last axis)
Bug desc  : CumSum(last axis) + erf + neg-abs-identity + residual-add-relu pattern.
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
from scipy.special import erf as sp_erf

np.random.seed(5)

# ── constants ─────────────────────────────────────────────────────────────────
B, C, H, W = 1, 2, 4, 4

res_val  = np.zeros((B, C, H, W), dtype=np.float32)
mm_w_val = (np.random.randn(1, 1, W, W) * 0.1).astype(np.float32)
cs_ax_val = np.array(3, dtype=np.int64)   # CumSum last axis

# ── model ─────────────────────────────────────────────────────────────────────
# X: [1, C, H, W]
# Add(residual=0) → Relu
# Tanh → Erf → Mul(relu_out, erf_out)   (erf-chain activation)
# Neg → Abs                              (neg-abs = identity)
# MatMul 4D
# CumSum axis=3 (last)
X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, C, H, W])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, C, H, W])

nodes = [
    oh.make_node('Add',    ['X', 'residual'], ['added']),
    oh.make_node('Relu',   ['added'],          ['relu_out']),
    oh.make_node('Tanh',   ['relu_out'],       ['tanh_out']),
    oh.make_node('Erf',    ['tanh_out'],       ['erf_out']),
    oh.make_node('Mul',    ['erf_out', 'relu_out'], ['act_out']),
    oh.make_node('Neg',    ['act_out'],        ['neg_out']),
    oh.make_node('Abs',    ['neg_out'],        ['abs_out']),
    oh.make_node('MatMul', ['abs_out', 'mm_w'], ['mm_out']),
    oh.make_node('CumSum', ['mm_out', 'cs_ax'], ['Y']),
]

inits = [
    nph.from_array(res_val,  name='residual'),
    nph.from_array(mm_w_val, name='mm_w'),
    nph.from_array(cs_ax_val, name='cs_ax'),
]

graph = oh.make_graph(nodes, 'bug_v2_0035', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.random.randn(B, C, H, W).astype(np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
added    = x + res_val
relu     = np.maximum(added, 0)
tanh_out = np.tanh(relu)
erf_out  = sp_erf(tanh_out)
act      = erf_out * relu
neg      = -act
abs_out  = np.abs(neg)            # neg-abs of non-negative = identity
mm       = abs_out @ mm_w_val[0, 0]
cs_np    = np.cumsum(mm, axis=3)  # CumSum last axis

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
all_vs_numpy = np.allclose(o_all, cs_np, atol=tol)

print(f"ORT_ENABLE_ALL  sample flat[0]: {o_all.ravel()[0]:.6f}")
print(f"ORT_DISABLE_ALL sample flat[0]: {o_dis.ravel()[0]:.6f}")
print(f"numpy reference sample flat[0]: {cs_np.ravel()[0]:.6f}")
print(f"ORT_ALL == ORT_DIS  : {all_vs_dis}")
print(f"ORT_ALL == numpy    : {all_vs_numpy}")
PASS = all_vs_numpy
print(f"PASS={PASS}")
