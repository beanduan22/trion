#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0016
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : n/a
Root cause : ORT optimizer eliminates the Unsqueeze/Squeeze identity chain after
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

C, H, W = 4, 8, 8

roi    = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), name='scales')

# Unsqueeze/Squeeze axes: add 3 dims then remove same 3 (identity, but tricks optimizer)
us_ax = numpy_helper.from_array(np.array([4, 5, 6], dtype=np.int64), name='us_ax')
sq_ax = numpy_helper.from_array(np.array([4, 5, 6], dtype=np.int64), name='sq_ax')

np.random.seed(5)
# After Resize [1,C,H,W]->[1,C,2H,2W], MatMul last dim: [1,1,2W,2W]
mm_w = numpy_helper.from_array(
    np.random.randn(1, 1, 2*W, 2*W).astype(np.float32) * 0.1, name='mm_w')

# ELU-variant (SELU-like): conv + ELU activation
conv_w = numpy_helper.from_array(
    np.random.randn(C, C, 3, 3).astype(np.float32) * 0.1, name='conv_w')
conv_b = numpy_helper.from_array(np.zeros(C, dtype=np.float32), name='conv_b')
alpha  = numpy_helper.from_array(np.array([1.6732632], dtype=np.float32), name='alpha')
lscale = numpy_helper.from_array(np.array([1.050701], dtype=np.float32), name='lscale')
one    = numpy_helper.from_array(np.array([1.], dtype=np.float32), name='one')
zero   = numpy_helper.from_array(np.array([0.], dtype=np.float32), name='zero')

# Final MatMul: [1,C,2H,2W] x [1,1,2W,2W]
mm_w2  = numpy_helper.from_array(
    np.random.randn(1, 1, 2*W, 2*W).astype(np.float32) * 0.1, name='mm_w2')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    helper.make_node('Resize', ['X', 'roi', 'scales'], ['resized'],
                     coordinate_transformation_mode='asymmetric',
                     mode='linear'),
    helper.make_node('Unsqueeze', ['resized', 'us_ax'], ['unsqz']),
    helper.make_node('Squeeze',   ['unsqz', 'sq_ax'],   ['sqz_out']),
    helper.make_node('MatMul', ['sqz_out', 'mm_w'], ['mm_out']),
    helper.make_node('Conv', ['mm_out', 'conv_w', 'conv_b'], ['cv_out'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    # ELU variant: max(x,0) + min(alpha*(exp(x)-1), 0)  * scale
    helper.make_node('Max',  ['cv_out', 'zero'], ['mx']),
    helper.make_node('Exp',  ['cv_out'], ['ex']),
    helper.make_node('Sub',  ['ex', 'one'], ['ex1']),
    helper.make_node('Mul',  ['ex1', 'alpha'], ['mu']),
    helper.make_node('Min',  ['mu', 'zero'], ['mn']),
    helper.make_node('Add',  ['mx', 'mn'], ['elu_sum']),
    helper.make_node('Mul',  ['elu_sum', 'lscale'], ['elu_out']),
    helper.make_node('MatMul', ['elu_out', 'mm_w2'], ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[roi, scales, us_ax, sq_ax, mm_w, conv_w, conv_b,
                 alpha, lscale, one, zero, mm_w2])
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
