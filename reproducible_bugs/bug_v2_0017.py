#!/usr/bin/env python3
"""
Bug v2-0017 — ORT log(exp(x))->x simplification + axis-0 softmax + mul-zero
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT optimizer cancels the Clip->Exp->Log sequence (treating it as
            an identity), but the Clip bounds are finite so the cancellation is
            only valid within those bounds. Combined with an axis-0 Softmax and
            a mul-by-zero+add-constant pattern, this causes divergence in the
            downstream LayerNorm-style normalization.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

D = 16  # reduced from 256 for speed

lo    = numpy_helper.from_array(np.array([-10.], dtype=np.float32), name='lo')
hi    = numpy_helper.from_array(np.array([10.], dtype=np.float32), name='hi')
scale = numpy_helper.from_array(np.ones([1, D], dtype=np.float32), name='scale')

zeros = numpy_helper.from_array(np.zeros([1, D], dtype=np.float32), name='zeros')
mze_s = numpy_helper.from_array(np.full([1, D], 1.05, dtype=np.float32), name='mze_s')
alpha = numpy_helper.from_array(np.full([1, D], 0.5, dtype=np.float32), name='alpha')

eps_v = numpy_helper.from_array(np.array([1e-5], dtype=np.float32), name='eps_v')
ln_g  = numpy_helper.from_array(np.ones(D, dtype=np.float32), name='ln_g')
ln_b  = numpy_helper.from_array(np.zeros(D, dtype=np.float32), name='ln_b')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, D])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    # log(exp(clip(x))) — ORT may cancel exp->log but clip bounds make it not a no-op
    helper.make_node('Clip', ['X', 'lo', 'hi'], ['clipped']),
    helper.make_node('Exp',  ['clipped'],        ['exp_out']),
    helper.make_node('Log',  ['exp_out'],         ['log_out']),
    helper.make_node('Mul',  ['log_out', 'scale'], ['lec_out']),
    # Softmax along axis=0 (unusual axis)
    helper.make_node('Softmax', ['lec_out'], ['sm_out'], axis=0),
    helper.make_node('Add', ['sm_out', 'lec_out'], ['sx_out']),
    # Mul-by-zero + add constant
    helper.make_node('Mul', ['sx_out', 'zeros'], ['mze_mz']),
    helper.make_node('Add', ['mze_mz', 'mze_s'], ['mze_out']),
    # Activation: neg -> abs -> relu -> scale
    helper.make_node('Neg',  ['mze_out'],  ['neg_out']),
    helper.make_node('Abs',  ['neg_out'],  ['abs_out']),
    helper.make_node('Relu', ['abs_out'],  ['relu_out']),
    helper.make_node('Mul',  ['relu_out', 'alpha'], ['anr_out']),
    # Manual LayerNorm
    helper.make_node('ReduceMean', ['anr_out'], ['mu'],  axes=[-1], keepdims=1),
    helper.make_node('Sub',  ['anr_out', 'mu'],  ['centered']),
    helper.make_node('Mul',  ['centered', 'centered'], ['sq']),
    helper.make_node('ReduceMean', ['sq'], ['var'], axes=[-1], keepdims=1),
    helper.make_node('Add',  ['var', 'eps_v'], ['var_eps']),
    helper.make_node('Sqrt', ['var_eps'],      ['std']),
    helper.make_node('Div',  ['centered', 'std'], ['normed']),
    helper.make_node('Mul',  ['normed', 'ln_g'],  ['scaled']),
    helper.make_node('Add',  ['scaled', 'ln_b'],  ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[lo, hi, scale, zeros, mze_s, alpha, eps_v, ln_g, ln_b])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
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
