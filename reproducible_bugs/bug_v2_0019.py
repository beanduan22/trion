#!/usr/bin/env python3
"""
Bug v2-0019 — ORT Resize(nearest_ceil) + log(exp(x))->x cancel + log-clamp
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT optimizer cancels the Clip->Exp->Log sequence after a log-clamp
            preprocessing step, but the combined clamping bounds of the two Clip
            ops are not correctly propagated, resulting in values outside the
            safe log domain being fed to downstream Ceil and GlobalAveragePool.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

C, H, W = 3, 8, 8
H2, W2 = 2*H, 2*W

roi    = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')
scales = numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), name='scales')

# log-clamp: abs(x) + eps -> log -> clip to [-5.5, 5.5]
lc_eps = numpy_helper.from_array(
    np.zeros([1, C, H2, W2], dtype=np.float32), name='lc_eps')
lc_min = numpy_helper.from_array(np.float32(-5.5), name='lc_min')
lc_max = numpy_helper.from_array(np.float32(5.5), name='lc_max')

# Second clip + exp + log (the cancel target)
lo    = numpy_helper.from_array(np.array([-10.], dtype=np.float32), name='lo')
hi    = numpy_helper.from_array(np.array([10.], dtype=np.float32), name='hi')
scale = numpy_helper.from_array(np.ones([1, C, H2, W2], dtype=np.float32), name='scale')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    # Nearest-ceil resize
    helper.make_node('Resize', ['X', 'roi', 'scales'], ['resized'],
                     coordinate_transformation_mode='half_pixel',
                     mode='nearest', nearest_mode='ceil'),
    # log-clamp block
    helper.make_node('Abs',  ['resized'],          ['abs_out']),
    helper.make_node('Add',  ['abs_out', 'lc_eps'], ['lc_add']),
    helper.make_node('Log',  ['lc_add'],            ['lc_log']),
    helper.make_node('Clip', ['lc_log', 'lc_min', 'lc_max'], ['lc_out']),
    # Clip -> Exp -> Log  (ORT may cancel to identity)
    helper.make_node('Clip', ['lc_out', 'lo', 'hi'], ['cl_out']),
    helper.make_node('Exp',  ['cl_out'],              ['ex_out']),
    helper.make_node('Log',  ['ex_out'],              ['lg_out']),
    helper.make_node('Mul',  ['lg_out', 'scale'],     ['lec_out']),
    # Ceil * x (ceil-multiply pattern)
    helper.make_node('Ceil', ['lec_out'], ['ceil_out']),
    helper.make_node('Mul',  ['ceil_out', 'lec_out'], ['cm_out']),
    # Channel attention via GlobalAveragePool + sigmoid
    helper.make_node('GlobalAveragePool', ['cm_out'], ['gap']),
    helper.make_node('Sigmoid', ['gap'], ['sig']),
    helper.make_node('Mul', ['cm_out', 'sig'], ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[roi, scales, lc_eps, lc_min, lc_max, lo, hi, scale])
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
