#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0026
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : MM(256->256) -> scale -> bias -> Cast(fp16) -> Cast(fp32) ->
Root cause : ORT's optimizer incorrectly eliminates the multiply-by-zero (mul-zero-elim pass)
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

B, T, D = 1, 32, 256
D_mid   = 64

mm_w    = rng.standard_normal((D, D)).astype(np.float32) * 0.05
mm_sc   = np.array([0.125], dtype=np.float32)
mm_bias = np.zeros((1, 1, D), dtype=np.float32)
zeros   = np.zeros((1, T, D), dtype=np.float32)
mze_sc  = np.full((1, T, D), 1.05, dtype=np.float32)
ln_sc1  = np.ones(D, dtype=np.float32)
ln_b1   = np.zeros(D, dtype=np.float32)
residual = np.zeros((1, T, D), dtype=np.float32)
w1      = rng.standard_normal((D, D_mid)).astype(np.float32) * 0.05
b1      = np.zeros(D_mid, dtype=np.float32)
w2      = rng.standard_normal((D_mid, D)).astype(np.float32) * 0.05
b2      = np.zeros(D, dtype=np.float32)
ln_sc2  = np.ones(D, dtype=np.float32)
ln_b2   = np.zeros(D, dtype=np.float32)

nodes = [
    helper.make_node('MatMul', ['X', 'mm_w'], ['mm_out']),
    helper.make_node('Mul',    ['mm_out', 'mm_sc'],   ['sc_out']),
    helper.make_node('Add',    ['sc_out', 'mm_bias'],  ['bias_out']),
    # cast roundtrip: fp32 -> fp16 -> fp32
    helper.make_node('Cast', ['bias_out'], ['fp16_out'], to=TensorProto.FLOAT16),
    helper.make_node('Cast', ['fp16_out'], ['fp32_out'], to=TensorProto.FLOAT),
    # mul-zero (should NOT be eliminated — the scale adds a constant offset)
    helper.make_node('Mul', ['fp32_out', 'zeros'],  ['mz_out']),
    helper.make_node('Add', ['mz_out',   'mze_sc'], ['mze_out']),
    # layer norm + residual
    helper.make_node('LayerNormalization', ['mze_out', 'ln_sc1', 'ln_b1'], ['ln1_out'],
                     axis=-1, epsilon=1e-5),
    helper.make_node('Add', ['ln1_out', 'residual'], ['res_out']),
    # channel attention (SENet) — axes as attribute (opset 13 style)
    helper.make_node('ReduceMean', ['res_out'], ['mu_out'], axes=[1], keepdims=1),
    helper.make_node('MatMul', ['mu_out', 'w1'], ['se1_out']),
    helper.make_node('Add',    ['se1_out', 'b1'], ['se1_ad']),
    helper.make_node('Relu',   ['se1_ad'],        ['se1_relu']),
    helper.make_node('MatMul', ['se1_relu', 'w2'], ['se2_out']),
    helper.make_node('Add',    ['se2_out', 'b2'], ['se2_ad']),
    helper.make_node('Sigmoid', ['se2_ad'],        ['se_sig']),
    helper.make_node('Mul', ['res_out', 'se_sig'], ['se_out']),
    helper.make_node('LayerNormalization', ['se_out', 'ln_sc2', 'ln_b2'], ['Y'],
                     axis=-1, epsilon=1e-5),
]

initializers = [
    numpy_helper.from_array(mm_w,     'mm_w'),
    numpy_helper.from_array(mm_sc,    'mm_sc'),
    numpy_helper.from_array(mm_bias,  'mm_bias'),
    numpy_helper.from_array(zeros,    'zeros'),
    numpy_helper.from_array(mze_sc,   'mze_sc'),
    numpy_helper.from_array(ln_sc1,   'ln_sc1'),
    numpy_helper.from_array(ln_b1,    'ln_b1'),
    numpy_helper.from_array(residual, 'residual'),
    numpy_helper.from_array(w1,       'w1'),
    numpy_helper.from_array(b1,       'b1'),
    numpy_helper.from_array(w2,       'w2'),
    numpy_helper.from_array(b2,       'b2'),
    numpy_helper.from_array(ln_sc2,   'ln_sc2'),
    numpy_helper.from_array(ln_b2,    'ln_b2'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0026',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [B, T, D])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [B, T, D])],
    initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8

INPUT = rng.standard_normal((B, T, D)).astype(np.float32)

sess_opt = ort.InferenceSession(model.SerializeToString(),
                                providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model.SerializeToString(), sess_options=opts,
                                providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

diff = float(np.max(np.abs(got - expected)))
print(f'max_diff={diff:.4e}')
print(f'PASS={diff < 1e-4}')

PASS = diff < 1e-4
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
