#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0023
Source     : Campaign v2 (fuzzing)
Compiler   : TF XLA (JAX GPU JIT) — reproduced here with OnnxRuntime ORT_ENABLE_ALL vs ORT_DISABLE_ALL
Patterns   : MM4D -> [Conv-BN path] + [Conv-scale-bias path] -> Add ->
Root cause : XLA GPU JIT fuses dual conv paths (one through BN, one through scale+bias) that share
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

C, H, W = 128, 16, 16

mm_w    = rng.standard_normal((1, 1, H, W)).astype(np.float32) * 0.2
conv_w  = rng.standard_normal((C, C, 3, 3)).astype(np.float32) * 0.05
conv_b  = np.zeros(C, dtype=np.float32)
bn_sc   = np.ones(C, dtype=np.float32)
bn_bias = np.zeros(C, dtype=np.float32)
bn_mean = np.zeros(C, dtype=np.float32)
bn_var  = np.ones(C, dtype=np.float32)
ch_sc   = np.ones((1, C, 1, 1), dtype=np.float32)
ch_bi   = np.zeros((1, C, 1, 1), dtype=np.float32)
flat_shape = np.array([1, C * H * W], dtype=np.int64)
back_shape = np.array([1, C, H, W], dtype=np.int64)
roi     = np.array([], dtype=np.float32)
scales  = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
czl_z   = np.zeros((1, 1, H * 2, W * 2), dtype=np.float32)

nodes = [
    # 4D MatMul (spatial)
    helper.make_node('MatMul', ['X', 'mm_w'],  ['mm_out']),
    # path 1: Conv -> BN
    helper.make_node('Conv', ['mm_out', 'conv_w', 'conv_b'], ['cv1_out'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node('BatchNormalization', ['cv1_out', 'bn_sc', 'bn_bias', 'bn_mean', 'bn_var'],
                     ['bn_out'], epsilon=1e-5),
    # path 2: Conv (shared weight) -> scale -> bias
    helper.make_node('Conv', ['mm_out', 'conv_w', 'conv_b'], ['cv2_out'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node('Mul', ['cv2_out', 'ch_sc'], ['mu_out']),
    helper.make_node('Add', ['mu_out', 'ch_bi'], ['ad_out']),
    # merge
    helper.make_node('Add', ['bn_out', 'ad_out'], ['merge_out']),
    # identity reshape round-trip x2
    helper.make_node('Reshape', ['merge_out', 'flat_shape'], ['rs1']),
    helper.make_node('Reshape', ['rs1',        'back_shape'], ['rs2']),
    helper.make_node('Reshape', ['rs2',         'flat_shape'], ['rs3']),
    helper.make_node('Reshape', ['rs3',         'back_shape'], ['rs4']),
    # resize align_corners linear (bug trigger)
    helper.make_node('Resize', ['rs4', 'roi', 'scales'], ['resi_out'],
                     coordinate_transformation_mode='align_corners',
                     mode='linear'),
    # concat zeros padding
    helper.make_node('Concat', ['resi_out', 'czl_z'], ['Y'], axis=1),
]

initializers = [
    numpy_helper.from_array(mm_w,       'mm_w'),
    numpy_helper.from_array(conv_w,     'conv_w'),
    numpy_helper.from_array(conv_b,     'conv_b'),
    numpy_helper.from_array(bn_sc,      'bn_sc'),
    numpy_helper.from_array(bn_bias,    'bn_bias'),
    numpy_helper.from_array(bn_mean,    'bn_mean'),
    numpy_helper.from_array(bn_var,     'bn_var'),
    numpy_helper.from_array(ch_sc,      'ch_sc'),
    numpy_helper.from_array(ch_bi,      'ch_bi'),
    numpy_helper.from_array(flat_shape, 'flat_shape'),
    numpy_helper.from_array(back_shape, 'back_shape'),
    numpy_helper.from_array(roi,        'roi'),
    numpy_helper.from_array(scales,     'scales'),
    numpy_helper.from_array(czl_z,      'czl_z'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0023',
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
