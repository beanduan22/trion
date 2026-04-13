#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0028
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : Input(1,3,32,32) -> Resize(nearest_ceil, half_pixel, 2x) ->
Root cause : ORT incorrectly fuses Resize(nearest_ceil) with a downstream ReduceSum that
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

C, H, W = 3, 32, 32
H2, W2  = H * 2, W * 2

roi    = np.array([], dtype=np.float32)
scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
rsm_ax = np.array([2], dtype=np.int64)
eps    = np.array([1e-6], dtype=np.float32)
# scale has shape (1,C,H2,W2) — same shape as resize output
msp_sc = np.ones((1, C, H2, W2), dtype=np.float32)
in_sc  = np.ones(C, dtype=np.float32)
in_b   = np.zeros(C, dtype=np.float32)

nodes = [
    # resize: (1,C,H,W) -> (1,C,H2,W2)
    helper.make_node('Resize', ['X', 'roi', 'scales'], ['r_out'],
                     coordinate_transformation_mode='half_pixel',
                     mode='nearest',
                     nearest_mode='ceil'),
    # reduce sum over height axis: (1,C,H2,W2) -> (1,C,1,W2) with keepdims=1
    helper.make_node('ReduceSum', ['r_out', 'rsm_ax'], ['rs_out'], keepdims=1),
    # add back to resize output (broadcasts over H2 dim)
    helper.make_node('Add', ['r_out', 'rs_out'], ['ad_out']),
    # sqrt(x^2 + eps) * scale (pseudo-RMS norm)
    helper.make_node('Mul',  ['ad_out', 'ad_out'], ['sq_out']),
    helper.make_node('Add',  ['sq_out', 'eps'],    ['sq_eps']),
    helper.make_node('Sqrt', ['sq_eps'],            ['sqrt_out']),
    helper.make_node('Mul',  ['sqrt_out', 'msp_sc'], ['msp_out']),
    # relu + residual relu
    helper.make_node('Relu', ['msp_out'],           ['relu1']),
    helper.make_node('Add',  ['relu1', 'msp_out'],  ['rar_ad']),
    helper.make_node('Relu', ['rar_ad'],             ['relu2']),
    # instance norm + final relu
    helper.make_node('InstanceNormalization', ['relu2', 'in_sc', 'in_b'], ['in_out'],
                     epsilon=1e-5),
    helper.make_node('Relu', ['in_out'], ['Y']),
]

initializers = [
    numpy_helper.from_array(roi,    'roi'),
    numpy_helper.from_array(scales, 'scales'),
    numpy_helper.from_array(rsm_ax, 'rsm_ax'),
    numpy_helper.from_array(eps,    'eps'),
    numpy_helper.from_array(msp_sc, 'msp_sc'),
    numpy_helper.from_array(in_sc,  'in_sc'),
    numpy_helper.from_array(in_b,   'in_b'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0028',
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
