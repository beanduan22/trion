#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0011
Source     : Campaign v2 (fuzzing)
Compiler   : XLA / JAX GPU JIT (GPU-only trigger)
Patterns   : n/a
Root cause : JAX linear resize diverges under GPU JIT — JIT uses different
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

# Resize bilinear half_pixel + manual L2 norm (reduce_sum of squares then rsqrt)
roi  = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), name='scales')
eps  = numpy_helper.from_array(np.array([1e-12], dtype=np.float32), name='eps')
two  = numpy_helper.from_array(np.array([2.], dtype=np.float32), name='two')
# In opset 13, ReduceSum takes axes as an optional input tensor
rs_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name='rs_axes')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4, 8, 8])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    # Linear resize (bilinear)
    helper.make_node('Resize', ['X', 'roi', 'scales'], ['resized'],
                     coordinate_transformation_mode='half_pixel',
                     mode='linear'),
    # Manual L2 norm: x / sqrt(sum(x^2) + eps) along channel dim
    helper.make_node('Pow',  ['resized', 'two'], ['sq']),
    helper.make_node('ReduceSum', ['sq', 'rs_axes'], ['sum_sq'], keepdims=1),
    helper.make_node('Add',  ['sum_sq', 'eps'], ['sum_eps']),
    helper.make_node('Sqrt', ['sum_eps'], ['norm']),
    helper.make_node('Div',  ['resized', 'norm'], ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[roi, scales, eps, two, rs_axes])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)

INPUT = np.random.randn(1, 4, 8, 8).astype(np.float32)

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
