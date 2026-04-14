#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0029
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : MM4D(shared_w) -> MM4D(shared_w) -> Resize(nearest_ceil, half_pixel, 2x) ->
Root cause : ORT's optimizer incorrectly reorders a Resize(nearest_ceil) with a downstream
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

C, H, W = 64, 32, 32
H2, W2  = H * 2, W * 2

mm_w   = rng.standard_normal((1, 1, H, W)).astype(np.float32) * 0.2
roi    = np.array([], dtype=np.float32)
scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

conv_w = rng.standard_normal((C, C, 3, 3)).astype(np.float32) * 0.05
conv_b = np.zeros(C, dtype=np.float32)

# GroupNorm: reshape (1,C,H2,W2) -> (1,G,C//G,H2,W2), reduce over dims [2,3,4]
G       = 8
rs1     = np.array([1, G, C // G, H2, W2], dtype=np.int64)
rs2     = np.array([1, C, H2, W2],          dtype=np.int64)
gn_eps  = np.array([1e-5], dtype=np.float32)
gn_g    = np.ones((1, C, 1, 1), dtype=np.float32)
gn_b    = np.zeros((1, C, 1, 1), dtype=np.float32)

nodes = [
    # two 4D batched matmuls with shared weight
    helper.make_node('MatMul', ['X',      'mm_w'], ['mm1']),
    helper.make_node('MatMul', ['mm1',    'mm_w'], ['mm2']),
    # resize nearest_ceil (bug trigger)
    helper.make_node('Resize', ['mm2', 'roi', 'scales'], ['r_out'],
                     coordinate_transformation_mode='half_pixel',
                     mode='nearest',
                     nearest_mode='ceil'),
    # multi-dilation conv branches
    helper.make_node('Conv', ['r_out', 'conv_w', 'conv_b'], ['c1'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                     dilations=[1, 1]),
    helper.make_node('Conv', ['r_out', 'conv_w', 'conv_b'], ['c2'],
                     kernel_shape=[3, 3], pads=[2, 2, 2, 2],
                     dilations=[2, 2]),
    helper.make_node('Conv', ['r_out', 'conv_w', 'conv_b'], ['c4'],
                     kernel_shape=[3, 3], pads=[4, 4, 4, 4],
                     dilations=[4, 4]),
    helper.make_node('Add', ['c1', 'c2'],      ['s1']),
    helper.make_node('Add', ['s1', 'c4'],      ['dbs_out']),
    # manual GroupNorm
    helper.make_node('Reshape', ['dbs_out', 'rs1'], ['gn_r1']),
    helper.make_node('ReduceMean', ['gn_r1'], ['gn_mu'],
                     axes=[2, 3, 4], keepdims=1),
    helper.make_node('Sub',  ['gn_r1', 'gn_mu'],    ['gn_sub']),
    helper.make_node('Mul',  ['gn_sub', 'gn_sub'],   ['gn_sq']),
    helper.make_node('ReduceMean', ['gn_sq'], ['gn_ms'],
                     axes=[2, 3, 4], keepdims=1),
    helper.make_node('Add',  ['gn_ms', 'gn_eps'],    ['gn_add']),
    helper.make_node('Sqrt', ['gn_add'],             ['gn_rt']),
    helper.make_node('Div',  ['gn_sub', 'gn_rt'],    ['gn_div']),
    helper.make_node('Reshape', ['gn_div', 'rs2'],   ['gn_r2']),
    helper.make_node('Mul',  ['gn_r2', 'gn_g'],      ['gn_sc']),
    helper.make_node('Add',  ['gn_sc', 'gn_b'],      ['gn_out']),
    # swish activation
    helper.make_node('Sigmoid', ['gn_out'],           ['sig']),
    helper.make_node('Mul',     ['gn_out', 'sig'],    ['Y']),
]

initializers = [
    numpy_helper.from_array(mm_w,   'mm_w'),
    numpy_helper.from_array(roi,    'roi'),
    numpy_helper.from_array(scales, 'scales'),
    numpy_helper.from_array(conv_w, 'conv_w'),
    numpy_helper.from_array(conv_b, 'conv_b'),
    numpy_helper.from_array(rs1,    'rs1'),
    numpy_helper.from_array(rs2,    'rs2'),
    numpy_helper.from_array(gn_eps, 'gn_eps'),
    numpy_helper.from_array(gn_g,   'gn_g'),
    numpy_helper.from_array(gn_b,   'gn_b'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0029',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, C, H2, W2])],
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
