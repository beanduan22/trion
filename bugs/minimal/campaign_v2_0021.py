#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0021
Source     : Campaign v2 (fuzzing)
Compiler   : TF XLA (JAX GPU JIT) — reproduced here with OnnxRuntime ORT_ENABLE_ALL vs ORT_DISABLE_ALL
Patterns   : DepthToSpace(CRD, block=2) -> Conv1x1 -> Resize(linear, align_corners) ->
Root cause : JAX/XLA GPU JIT miscompiles a pipeline that pixel-shuffles via DepthToSpace (CRD mode),
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

# DepthToSpace: block_size=2, CRD mode: (1,64,32,32) -> (1,16,64,64)
block = 2
C_in, H_in, W_in = 64, 32, 32
C_d2s = C_in // (block * block)  # 16
H_d2s, W_d2s = H_in * block, W_in * block  # 64, 64

# Conv 1x1: (1,16,64,64) -> (1,32,64,64)
C_conv = 32
conv_w = rng.standard_normal((C_conv, C_d2s, 1, 1)).astype(np.float32) * 0.3
conv_b = np.zeros(C_conv, dtype=np.float32)

# Resize align_corners, linear: (1,32,64,64) -> (1,32,128,128)
roi    = np.array([], dtype=np.float32)
scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

nodes = [
    helper.make_node('DepthToSpace', ['X'], ['d2s_out'],
                     blocksize=block, mode='CRD'),
    helper.make_node('Conv', ['d2s_out', 'conv_w', 'conv_b'], ['conv_out'],
                     kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
    helper.make_node('Resize', ['conv_out', 'roi', 'scales'], ['resi_out'],
                     coordinate_transformation_mode='align_corners',
                     mode='linear'),
    helper.make_node('Transpose', ['resi_out'], ['tr1_out'],
                     perm=[0, 2, 3, 1]),
    helper.make_node('Transpose', ['tr1_out'], ['tr2_out'],
                     perm=[0, 3, 1, 2]),
    helper.make_node('Relu', ['tr2_out'], ['Y']),
]

initializers = [
    numpy_helper.from_array(conv_w, 'conv_w'),
    numpy_helper.from_array(conv_b, 'conv_b'),
    numpy_helper.from_array(roi,    'roi'),
    numpy_helper.from_array(scales, 'scales'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0021',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C_in, H_in, W_in])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)],
    initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8

INPUT = rng.standard_normal((1, C_in, H_in, W_in)).astype(np.float32)

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
