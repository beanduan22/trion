#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0010
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : n/a
Root cause : ORT optimizer mishandles Abs+Pow(2)+Sqrt chain after nearest_ceil
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

# Initializers
roi = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), name='scales')
two = numpy_helper.from_array(np.array([2.], dtype=np.float32), name='two')
eps = numpy_helper.from_array(np.array([1e-6], dtype=np.float32), name='eps')
# Small conv weights for speed
conv_w = numpy_helper.from_array(
    np.random.randn(8, 4, 3, 3).astype(np.float32) * 0.1, name='conv_w')
conv_b = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name='conv_b')
# BN params
bn_scale = numpy_helper.from_array(np.ones(8, dtype=np.float32), name='bn_scale')
bn_bias  = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name='bn_bias')
bn_mean  = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name='bn_mean')
bn_var   = numpy_helper.from_array(np.ones(8, dtype=np.float32), name='bn_var')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4, 8, 8])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    helper.make_node('Resize', ['X', 'roi', 'scales'], ['resized'],
                     coordinate_transformation_mode='half_pixel',
                     mode='nearest', nearest_mode='ceil'),
    helper.make_node('Abs',  ['resized'], ['abs_out']),
    helper.make_node('Add',  ['abs_out', 'eps'], ['add_out']),
    helper.make_node('Pow',  ['add_out', 'two'], ['pow_out']),
    helper.make_node('Sqrt', ['pow_out'], ['sqrt_out']),
    helper.make_node('Conv', ['sqrt_out', 'conv_w', 'conv_b'], ['conv_out'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
    helper.make_node('BatchNormalization',
                     ['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
                     ['Y'], epsilon=1e-5),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[roi, scales, two, eps, conv_w, conv_b,
                 bn_scale, bn_bias, bn_mean, bn_var])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)

INPUT = np.random.randn(1, 4, 8, 8).astype(np.float32)

# Optimized run
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
got = sess.run(None, {'X': INPUT})[0]

# Reference (no optimizations)
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
