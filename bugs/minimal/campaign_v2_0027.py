#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0027
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : Input -> Conv(w1,b) -> Relu -> Conv(w1,b) -> Add(input) ->   [residual]
Root cause : ORT incorrectly fuses a residual double-Conv block (both branches sharing the same
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

C, H, W = 32, 32, 32

conv_w  = rng.standard_normal((C, C, 3, 3)).astype(np.float32) * 0.05
conv_b  = np.zeros(C, dtype=np.float32)
roi     = np.array([], dtype=np.float32)
scales  = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
# after resize: (1,32,64,64)
mm_w    = rng.standard_normal((1, 1, 64, 64)).astype(np.float32) * 0.2
sl_st1  = np.array([0, 0,  0, 0], dtype=np.int64)
sl_en1  = np.array([1, 16, 64, 64], dtype=np.int64)
sl_st2  = np.array([0, 16, 0, 0], dtype=np.int64)
sl_en2  = np.array([1, 32, 64, 64], dtype=np.int64)

nodes = [
    # residual conv block (shared weights)
    helper.make_node('Conv', ['X', 'conv_w', 'conv_b'], ['c1_out'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node('Relu', ['c1_out'], ['r1_out']),
    helper.make_node('Conv', ['r1_out', 'conv_w', 'conv_b'], ['c2_out'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node('Add',  ['c2_out', 'X'], ['res_out']),
    # resize asymmetric linear (bug trigger)
    helper.make_node('Resize', ['res_out', 'roi', 'scales'], ['resi_out'],
                     coordinate_transformation_mode='asymmetric',
                     mode='linear'),
    # swish = x * sigmoid(mm(x))
    helper.make_node('MatMul',  ['resi_out', 'mm_w'],  ['mm_out']),
    helper.make_node('Sigmoid', ['mm_out'],             ['sig_out']),
    helper.make_node('Mul',     ['mm_out', 'sig_out'],  ['swish_out']),
    # split into two halves along channel dim and concat
    helper.make_node('Slice', ['swish_out', 'sl_st1', 'sl_en1'], ['sa_out']),
    helper.make_node('Slice', ['swish_out', 'sl_st2', 'sl_en2'], ['sb_out']),
    helper.make_node('Concat', ['sa_out', 'sb_out'], ['Y'], axis=1),
]

initializers = [
    numpy_helper.from_array(conv_w, 'conv_w'),
    numpy_helper.from_array(conv_b, 'conv_b'),
    numpy_helper.from_array(roi,    'roi'),
    numpy_helper.from_array(scales, 'scales'),
    numpy_helper.from_array(mm_w,   'mm_w'),
    numpy_helper.from_array(sl_st1, 'sl_st1'),
    numpy_helper.from_array(sl_en1, 'sl_en1'),
    numpy_helper.from_array(sl_st2, 'sl_st2'),
    numpy_helper.from_array(sl_en2, 'sl_en2'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0027',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)],
    initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8

INPUT = rng.standard_normal((1, C, H, W)).astype(np.float32)

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
