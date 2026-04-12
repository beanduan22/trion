"""
Bug v2-0025 — TF XLA: Where(const_true) + depthwise Conv + Resize(linear, align_corners)
Compiler  : TF XLA (JAX GPU JIT) — reproduced here with OnnxRuntime ORT_ENABLE_ALL vs ORT_DISABLE_ALL
Root cause: XLA folds the Where node (condition is all-True constant) and then incorrectly
            constant-propagates through the depthwise convolution path into the final
            Resize(linear, align_corners) upsampler, producing wrong boundary pixel values.
Pattern   : Where(all_true, x, 0) -> Mul(x) ->
            Conv(3x3) -> Softplus ->
            DepthwiseConv(3x3) -> Tanh -> Cos -> Mul ->
            Resize(linear, align_corners)
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

C_in, H, W = 32, 16, 16

cond  = np.ones((1, C_in, H, W), dtype=bool)
zero  = np.zeros((1, C_in, H, W), dtype=np.float32)
C_mid = 64
conv_w  = rng.standard_normal((C_mid, C_in, 3, 3)).astype(np.float32) * 0.1
conv_b  = np.zeros(C_mid, dtype=np.float32)
# depthwise: groups=C_mid, weight shape (C_mid, 1, 3, 3)
dw_w    = np.full((C_mid, 1, 3, 3), 0.1, dtype=np.float32)
dw_b    = np.zeros(C_mid, dtype=np.float32)
roi     = np.array([], dtype=np.float32)
scales  = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

nodes = [
    # Where fold: condition is all-True constant, selects x
    helper.make_node('Where', ['cond', 'X', 'zero'], ['wh_out']),
    helper.make_node('Mul',   ['wh_out', 'X'],       ['mul_out']),
    # regular conv -> softplus activation
    helper.make_node('Conv', ['mul_out', 'conv_w', 'conv_b'], ['cv_out'],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node('Softplus', ['cv_out'], ['sp_out']),
    # depthwise conv (group=C_mid)
    helper.make_node('Conv', ['sp_out', 'dw_w', 'dw_b'], ['dw_out'],
                     group=C_mid, kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    # tanh -> cos -> gate mul
    helper.make_node('Tanh', ['dw_out'],           ['th_out']),
    helper.make_node('Cos',  ['th_out'],            ['cos_out']),
    helper.make_node('Mul',  ['cos_out', 'dw_out'], ['gate_out']),
    # resize align_corners linear (bug trigger)
    helper.make_node('Resize', ['gate_out', 'roi', 'scales'], ['Y'],
                     coordinate_transformation_mode='align_corners',
                     mode='linear'),
]

initializers = [
    numpy_helper.from_array(cond,   'cond'),
    numpy_helper.from_array(zero,   'zero'),
    numpy_helper.from_array(conv_w, 'conv_w'),
    numpy_helper.from_array(conv_b, 'conv_b'),
    numpy_helper.from_array(dw_w,   'dw_w'),
    numpy_helper.from_array(dw_b,   'dw_b'),
    numpy_helper.from_array(roi,    'roi'),
    numpy_helper.from_array(scales, 'scales'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0025',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C_in, H, W])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)],
    initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8

INPUT = rng.standard_normal((1, C_in, H, W)).astype(np.float32)

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
