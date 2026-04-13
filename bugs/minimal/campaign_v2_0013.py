#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0013
Source     : Campaign v2 (fuzzing)
Compiler   : XLA / JAX GPU JIT (GPU-only trigger)
Patterns   : n/a
Root cause : JAX linear resize with align_corners gets fused with group norm under
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

C, G = 4, 2   # channels, groups
H, W = 8, 8

roi    = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), name='scales')
eps_v  = numpy_helper.from_array(np.array([1e-5], dtype=np.float32), name='eps_v')
# gn_sc/gn_b shaped [1, C, 1, 1] for broadcast with [1, C, 2H, 2W]
gn_sc  = numpy_helper.from_array(np.ones([1, C, 1, 1], dtype=np.float32), name='gn_sc')
gn_b   = numpy_helper.from_array(np.zeros([1, C, 1, 1], dtype=np.float32), name='gn_b')

# Reshape constants: [1,C,2H,2W] -> [1,G,C//G*2H*2W] for group stats
shape_grp  = numpy_helper.from_array(
    np.array([1, G, C // G * 2*H * 2*W], dtype=np.int64), name='shape_grp')
shape_back = numpy_helper.from_array(
    np.array([1, C, 2*H, 2*W], dtype=np.int64), name='shape_back')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    helper.make_node('Resize', ['X', 'roi', 'scales'], ['resized'],
                     coordinate_transformation_mode='asymmetric',
                     mode='linear'),
    # Group norm: reshape -> per-group mean/var -> normalize -> reshape back
    helper.make_node('Reshape', ['resized', 'shape_grp'], ['grp']),
    helper.make_node('ReduceMean', ['grp'], ['grp_mean'],
                     axes=[2], keepdims=1),
    helper.make_node('Sub',  ['grp', 'grp_mean'], ['grp_centered']),
    helper.make_node('Mul',  ['grp_centered', 'grp_centered'], ['grp_sq']),
    helper.make_node('ReduceMean', ['grp_sq'], ['grp_var'],
                     axes=[2], keepdims=1),
    helper.make_node('Add',  ['grp_var', 'eps_v'], ['grp_var_eps']),
    helper.make_node('Sqrt', ['grp_var_eps'], ['grp_std']),
    helper.make_node('Div',  ['grp_centered', 'grp_std'], ['grp_norm']),
    helper.make_node('Reshape', ['grp_norm', 'shape_back'], ['normed']),
    # gn_sc/gn_b are [1,C,1,1] — broadcast correctly with [1,C,2H,2W]
    helper.make_node('Mul',  ['normed', 'gn_sc'], ['scaled']),
    helper.make_node('Add',  ['scaled', 'gn_b'], ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[roi, scales, eps_v, gn_sc, gn_b, shape_grp, shape_back])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)

INPUT = np.random.randn(1, C, H, W).astype(np.float32)

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
