#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0014
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : n/a
Root cause : ORT optimizer incorrectly simplifies the CumSum output when it
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

D = 16   # reduced from 64 for speed

# CumSum axis (last dim of [1, D])
cs_ax = numpy_helper.from_array(np.int64(1), name='cs_ax')

# Gemm weights
gemm_w = numpy_helper.from_array(
    (np.random.randn(D, D) * 0.1).astype(np.float32), name='gemm_w')
gemm_b = numpy_helper.from_array(np.zeros(D, dtype=np.float32), name='gemm_b')

# GELU expansion: D -> 4D
expand_w = numpy_helper.from_array(
    (np.random.randn(D, 4*D) * 0.1).astype(np.float32), name='expand_w')
expand_b = numpy_helper.from_array(np.zeros(4*D, dtype=np.float32), name='expand_b')

ln_sc  = numpy_helper.from_array(np.ones(4*D, dtype=np.float32), name='ln_sc')
ln_b   = numpy_helper.from_array(np.zeros(4*D, dtype=np.float32), name='ln_b')
sqrt2  = numpy_helper.from_array(np.array([1.4142135], dtype=np.float32), name='sqrt2')
half   = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name='half')
one    = numpy_helper.from_array(np.array([1.], dtype=np.float32), name='one')

# Gated GLU weights
gate_w1 = numpy_helper.from_array(
    (np.random.randn(4*D, 4*D) * 0.1).astype(np.float32), name='gate_w1')
gate_w2 = numpy_helper.from_array(
    (np.random.randn(4*D, 4*D) * 0.1).astype(np.float32), name='gate_w2')
gate_b  = numpy_helper.from_array(np.zeros(4*D, dtype=np.float32), name='gate_b')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, D])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    # CumSum over last axis
    helper.make_node('CumSum', ['X', 'cs_ax'], ['cs_out']),
    # Gemm + tanh
    helper.make_node('Gemm', ['cs_out', 'gemm_w', 'gemm_b'], ['gemm_out'],
                     alpha=1.0, beta=1.0, transB=0),
    helper.make_node('Tanh', ['gemm_out'], ['tanh_out']),
    # Expand + LayerNorm
    helper.make_node('MatMul', ['tanh_out', 'expand_w'], ['mm_out']),
    helper.make_node('Add',    ['mm_out', 'expand_b'],   ['mm_add']),
    helper.make_node('LayerNormalization',
                     ['mm_add', 'ln_sc', 'ln_b'], ['ln_out'],
                     axis=-1, epsilon=1e-5),
    # GELU: x * 0.5 * (1 + erf(x/sqrt(2)))
    helper.make_node('Div',  ['ln_out', 'sqrt2'], ['ln_div']),
    helper.make_node('Erf',  ['ln_div'], ['ln_erf']),
    helper.make_node('Add',  ['ln_erf', 'one'], ['ln_erf1']),
    helper.make_node('Mul',  ['ln_out', 'half'], ['ln_half']),
    helper.make_node('Mul',  ['ln_half', 'ln_erf1'], ['gelu_out']),
    # Gated GLU
    helper.make_node('Gemm', ['gelu_out', 'gate_w1', 'gate_b'], ['g1'],
                     alpha=1.0, beta=1.0, transB=0),
    helper.make_node('Gemm', ['gelu_out', 'gate_w2', 'gate_b'], ['g2'],
                     alpha=1.0, beta=1.0, transB=0),
    helper.make_node('Sigmoid', ['g2'], ['g2_sig']),
    helper.make_node('Mul', ['g1', 'g2_sig'], ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[cs_ax, gemm_w, gemm_b, expand_w, expand_b,
                 ln_sc, ln_b, sqrt2, half, one,
                 gate_w1, gate_w2, gate_b])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)

INPUT = np.random.randn(1, D).astype(np.float32)

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

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
