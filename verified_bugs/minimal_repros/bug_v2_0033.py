#!/usr/bin/env python3
"""
Minimal repro for bug v2-0033 (uid=0033).
Compiler  : OnnxRuntime
Patterns  : CumSum(last axis) + CBAM spatial attention + Softmax(axis=0) + mul_by_one chain + 4D matmul
Bug desc  : SUSPECT — original divergence may be a pytorch_eager (onnx2torch) reference error,
            not an ORT bug. ORT_ENABLE_ALL == ORT_DISABLE_ALL in the original model.
Oracle    : ORT_ENABLE_ALL vs ORT_DISABLE_ALL and numpy reference.
"""
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto
from onnx import shape_inference
import onnxruntime as ort

np.random.seed(3)

# ── constants ─────────────────────────────────────────────────────────────────
B, C, H, W = 1, 4, 4, 4

cbam_w_val = (np.random.randn(1, 2, 3, 3) * 0.05).astype(np.float32)
cbam_b_val = np.zeros(1, dtype=np.float32)
sal_w_val  = (np.random.randn(C, C, 1, 1) * 0.1).astype(np.float32)
sal_b_val  = np.zeros(C, dtype=np.float32)
one_val    = np.array([1.0], dtype=np.float32)
mm4d_w_val = (np.random.randn(1, 1, W, W) * 0.1).astype(np.float32)
cs_ax_val  = np.array(3, dtype=np.int64)   # CumSum on last axis
sq_ax_val  = np.array([0], dtype=np.int64) # Unsqueeze to add batch dim before Conv

# ── model ─────────────────────────────────────────────────────────────────────
X = oh.make_tensor_value_info('X', TensorProto.FLOAT, [B, C, H, W])
Y = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [B, C, H, W])

nodes = [
    oh.make_node('CumSum',    ['X', 'cs_ax'],          ['cs_out']),
    oh.make_node('ReduceMean',['cs_out'],               ['avg'], axes=[1], keepdims=0),
    oh.make_node('ReduceMax', ['cs_out'],               ['mx'],  axes=[1], keepdims=0),
    oh.make_node('Concat',    ['avg', 'mx'],            ['cat'],   axis=0),
    oh.make_node('Unsqueeze', ['cat', 'sq_ax'],         ['cat4d']),
    oh.make_node('Conv',      ['cat4d', 'cbam_w', 'cbam_b'], ['cv'],
                 kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    oh.make_node('Sigmoid',   ['cv'],                   ['sig']),
    oh.make_node('Mul',       ['cs_out', 'sig'],        ['cbam_out']),
    oh.make_node('Conv',      ['cbam_out', 'sal_w', 'sal_b'], ['sal_cv'],
                 kernel_shape=[1, 1]),
    oh.make_node('Relu',      ['sal_cv'],               ['sal_rl']),
    oh.make_node('Softmax',   ['sal_rl'],               ['sal_out'], axis=0),
    oh.make_node('Mul',       ['sal_out', 'one'],       ['m1']),
    oh.make_node('Mul',       ['m1',      'one'],       ['m2']),
    oh.make_node('Mul',       ['m2',      'one'],       ['m3']),
    oh.make_node('MatMul',    ['m3', 'mm4d_w'],         ['Y']),
]

inits = [
    nph.from_array(cs_ax_val,  name='cs_ax'),
    nph.from_array(cbam_w_val, name='cbam_w'),
    nph.from_array(cbam_b_val, name='cbam_b'),
    nph.from_array(sal_w_val,  name='sal_w'),
    nph.from_array(sal_b_val,  name='sal_b'),
    nph.from_array(one_val,    name='one'),
    nph.from_array(mm4d_w_val, name='mm4d_w'),
    nph.from_array(sq_ax_val,  name='sq_ax'),
]

graph = oh.make_graph(nodes, 'bug_v2_0033', [X], [Y], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)
model_bytes = model.SerializeToString()

# ── input ─────────────────────────────────────────────────────────────────────
x = np.random.randn(B, C, H, W).astype(np.float32)

# ── numpy oracle ──────────────────────────────────────────────────────────────
def conv2d_np(inp, w, b, pad=0):
    """Minimal conv2d for NCHW inputs."""
    N, Ci, IH, IW = inp.shape
    Co, _, Kh, Kw = w.shape
    OH = IH - Kh + 2*pad + 1
    OW = IW - Kw + 2*pad + 1
    padded = np.pad(inp, ((0,0),(0,0),(pad,pad),(pad,pad)))
    out = np.zeros((N, Co, OH, OW), dtype=np.float64)
    for co in range(Co):
        for ci in range(Ci):
            for i in range(OH):
                for j in range(OW):
                    out[:, co, i, j] += (padded[:, ci, i:i+Kh, j:j+Kw] * w[co, ci]).sum((-1,-2))
        out[:, co] += b[co]
    return out.astype(np.float32)

cs      = np.cumsum(x, axis=3)
avg     = cs.mean(axis=1)               # [1, H, W]
mx      = cs.max(axis=1)                # [1, H, W]
cat4d   = np.concatenate([avg, mx], axis=0)[np.newaxis]  # [1, 2, H, W]
cv      = conv2d_np(cat4d, cbam_w_val, cbam_b_val, pad=1)
sig     = 1.0 / (1.0 + np.exp(-cv))
cbam    = cs * sig
sal_cv  = conv2d_np(cbam, sal_w_val, sal_b_val, pad=0)
sal_rl  = np.maximum(sal_cv, 0)
e       = np.exp(sal_rl - sal_rl.max(axis=0, keepdims=True))
sal_out = e / e.sum(axis=0, keepdims=True)  # Softmax axis=0
m3      = sal_out                            # *1*1*1 = identity
out_np  = m3 @ mm4d_w_val[0, 0]

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
# SUSPECT: if PASS=False, verify whether pytorch_eager agrees with ORT or numpy
PASS = all_vs_numpy
print(f"PASS={PASS}")
