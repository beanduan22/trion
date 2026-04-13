#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0018
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : n/a
Root cause : ORT optimizer fuses the double Transpose-MatMul-Transpose pattern
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

# Reduced dimensions for speed: [1, 8, 16] instead of [1, 64, 128]
B, T, D = 1, 8, 16

np.random.seed(3)
w0    = numpy_helper.from_array(
    (np.random.randn(D, D) * 0.1).astype(np.float32), name='w0')
temp  = numpy_helper.from_array(np.array([0.505], dtype=np.float32), name='temp')
bias0 = numpy_helper.from_array(np.zeros([B, T, D], dtype=np.float32), name='bias0')
cs_ax = numpy_helper.from_array(np.int64(2), name='cs_ax')
wq    = numpy_helper.from_array(
    (np.random.randn(D, D) * 0.1).astype(np.float32), name='wq')
wk    = numpy_helper.from_array(
    (np.random.randn(D, D) * 0.1).astype(np.float32), name='wk')
wv    = numpy_helper.from_array(
    (np.random.randn(D, D) * 0.1).astype(np.float32), name='wv')
attn_scale = numpy_helper.from_array(
    np.array([1.0 / (D ** 0.5)], dtype=np.float32), name='attn_scale')
# ALiBi bias: lower-triangular negative distances
alibi = np.zeros([B, T, T], dtype=np.float32)
for i in range(T):
    for j in range(T):
        if j <= i:
            alibi[0, i, j] = -(i - j) * 0.125
        else:
            alibi[0, i, j] = -7.875
alibi_b = numpy_helper.from_array(alibi, name='alibi_b')
w_out = numpy_helper.from_array(
    (np.random.randn(D, D) * 0.1).astype(np.float32), name='w_out')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [B, T, D])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    # transpose-matmul-transpose pattern (double-transpose = identity, but tricks ORT)
    helper.make_node('Transpose', ['X'],        ['tr1'],  perm=[0, 2, 1]),
    helper.make_node('Transpose', ['tr1'],      ['tr2'],  perm=[0, 2, 1]),
    helper.make_node('MatMul',    ['tr2', 'w0'],['mm0']),
    helper.make_node('Transpose', ['mm0'],      ['tr3'],  perm=[0, 2, 1]),
    helper.make_node('Transpose', ['tr3'],      ['tmmt'], perm=[0, 2, 1]),
    # temperature scaling + bias
    helper.make_node('Div', ['tmmt', 'temp'],   ['div_out']),
    helper.make_node('Add', ['div_out', 'bias0'], ['lts_out']),
    # CumSum along last axis
    helper.make_node('CumSum', ['lts_out', 'cs_ax'], ['cs_out']),
    # Q, K, V projections
    helper.make_node('MatMul', ['cs_out', 'wq'], ['Q']),
    helper.make_node('MatMul', ['cs_out', 'wk'], ['K']),
    helper.make_node('MatMul', ['cs_out', 'wv'], ['V']),
    # Attention: softmax((Q @ K^T) * scale + alibi_b) @ V
    helper.make_node('Transpose', ['K'], ['Kt'], perm=[0, 2, 1]),
    helper.make_node('MatMul', ['Q', 'Kt'], ['raw_attn']),
    helper.make_node('Mul',    ['raw_attn', 'attn_scale'], ['scaled_attn']),
    helper.make_node('Add',    ['scaled_attn', 'alibi_b'], ['biased_attn']),
    helper.make_node('Softmax', ['biased_attn'], ['sm_attn'], axis=-1),
    helper.make_node('MatMul', ['sm_attn', 'V'], ['attn_out']),
    # output transpose-matmul-transpose
    helper.make_node('Transpose', ['attn_out'], ['otr1'],   perm=[0, 2, 1]),
    helper.make_node('Transpose', ['otr1'],     ['otr2'],   perm=[0, 2, 1]),
    helper.make_node('MatMul',    ['otr2', 'w_out'], ['omm']),
    helper.make_node('Transpose', ['omm'],      ['otr3'],   perm=[0, 2, 1]),
    helper.make_node('Transpose', ['otr3'],     ['Y'],      perm=[0, 2, 1]),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[w0, temp, bias0, cs_ax, wq, wk, wv,
                 attn_scale, alibi_b, w_out])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)

INPUT = np.random.randn(B, T, D).astype(np.float32)

sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
got = sess.run(None, {'X': INPUT})[0]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model.SerializeToString(), sess_options=opts,
                                providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

diff = float(np.max(np.abs(got - expected))) if got.shape == expected.shape else float('inf')
print(f'ORT_ENABLE_ALL:  {got.ravel()[:4]}')
print(f'ORT_DISABLE_ALL: {expected.ravel()[:4]}')
print(f'max_diff={diff:.4e}')
print(f'PASS={diff < 1e-4}')

PASS = diff < 1e-4
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
