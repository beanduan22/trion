#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0015
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : n/a
Root cause : ORT optimizer fuses the Softsign -> Mul -> Where gating pattern
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

C, H, W = 4, 8, 8

# Softsign: x / (1 + |x|), then Mul * 1.0, then Where(x > 0, x, 0)
soft_one = numpy_helper.from_array(np.array([1.], dtype=np.float32), name='soft_one')
thresh   = numpy_helper.from_array(np.array([0.], dtype=np.float32), name='thresh')
zeros    = numpy_helper.from_array(np.zeros([1, C, H, W], dtype=np.float32), name='zeros')

np.random.seed(11)
mm_w = numpy_helper.from_array(
    np.random.randn(1, 1, W, W).astype(np.float32) * 0.2, name='mm_w')

roi    = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), name='scales')
two_t  = numpy_helper.from_array(
    np.full([1, C, 2*H, 2*W], 2., dtype=np.float32), name='two_t')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    helper.make_node('Softsign', ['X'], ['soft_act']),
    helper.make_node('Mul',  ['soft_act', 'soft_one'], ['soft_out']),
    helper.make_node('Greater', ['soft_out', 'thresh'], ['gt_mask']),
    helper.make_node('Where',   ['gt_mask', 'soft_out', 'zeros'], ['gated']),
    helper.make_node('MatMul',  ['gated', 'mm_w'], ['mm_out']),
    helper.make_node('Resize',  ['mm_out', 'roi', 'scales'], ['resized'],
                     coordinate_transformation_mode='half_pixel',
                     mode='nearest', nearest_mode='ceil'),
    # asd pattern: out + out + out * two
    helper.make_node('Add', ['resized', 'resized'], ['add2']),
    helper.make_node('Mul', ['resized', 'two_t'],   ['mul2']),
    helper.make_node('Add', ['add2', 'mul2'], ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[soft_one, thresh, zeros, mm_w, roi, scales, two_t])
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

PASS = diff < 1e-4
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
