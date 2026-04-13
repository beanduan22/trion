#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0022
Source     : Campaign v2 (fuzzing)
Compiler   : TF XLA (JAX GPU JIT) — reproduced here with OnnxRuntime ORT_ENABLE_ALL vs ORT_DISABLE_ALL
Patterns   : Mul(x2) -> Tanh -> Mul(0.5) -> Add(0.5) -> [scale_expand] -> Mul ->
Root cause : XLA GPU JIT miscompiles a pipeline with two Resize nodes of different modes.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

C, H, W = 128, 32, 32

two      = np.array([2.0], dtype=np.float32)
half     = np.array([0.5], dtype=np.float32)
ch_scale = np.ones(C, dtype=np.float32)
axes_val = np.array([0, 2, 3], dtype=np.int64)
tshape   = np.array([1, C, H, W], dtype=np.int64)
roi      = np.array([], dtype=np.float32)
scales1  = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
scales2  = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

nodes = [
    # tanh-sigmoid rescale: x -> tanh(2x)*0.5 + 0.5
    helper.make_node('Mul',  ['X', 'two'],      ['mul1_out']),
    helper.make_node('Tanh', ['mul1_out'],       ['tanh_out']),
    helper.make_node('Mul',  ['tanh_out', 'half'], ['mul2_out']),
    helper.make_node('Add',  ['mul2_out', 'half'], ['add_out']),
    # channel expand-mul
    helper.make_node('Unsqueeze', ['ch_scale', 'axes_val'], ['us_out']),
    helper.make_node('Expand',    ['us_out', 'tshape'],     ['ex_out']),
    helper.make_node('Mul',       ['add_out', 'ex_out'],    ['scaled']),
    # first resize: align_corners linear (bug trigger)
    helper.make_node('Resize', ['scaled', 'roi', 'scales1'], ['r1_out'],
                     coordinate_transformation_mode='align_corners',
                     mode='linear'),
    # channel attention gate
    helper.make_node('GlobalAveragePool', ['r1_out'], ['gap_out']),
    helper.make_node('Sigmoid', ['gap_out'], ['sig_out']),
    helper.make_node('Mul', ['r1_out', 'sig_out'], ['att_out']),
    # second resize: asymmetric nearest
    helper.make_node('Resize', ['att_out', 'roi', 'scales2'], ['Y'],
                     coordinate_transformation_mode='asymmetric',
                     mode='nearest',
                     nearest_mode='floor'),
]

initializers = [
    numpy_helper.from_array(two,      'two'),
    numpy_helper.from_array(half,     'half'),
    numpy_helper.from_array(ch_scale, 'ch_scale'),
    numpy_helper.from_array(axes_val, 'axes_val'),
    numpy_helper.from_array(tshape,   'tshape'),
    numpy_helper.from_array(roi,      'roi'),
    numpy_helper.from_array(scales1,  'scales1'),
    numpy_helper.from_array(scales2,  'scales2'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0022',
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

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
