#!/usr/bin/env python3
"""
Bug v2-0012 — ORT mul_zero_elim + InstanceNormalization
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT optimizer eliminates the Mul-by-zero + Add-constant pattern
            (replacing the tensor with a constant), but this changes the input
            to InstanceNormalization, producing incorrect normalized results
            when the optimizer-produced constant does not match runtime values.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

# mul_zero_elim: X * zeros + scale  => should be scale (constant)
# but InstanceNorm depends on the actual channel statistics
zeros = numpy_helper.from_array(np.zeros([1, 3, 8, 8], dtype=np.float32), name='zeros')
scale_const = numpy_helper.from_array(
    np.full([1, 3, 8, 8], 1.05, dtype=np.float32), name='scale_const')
inn_sc = numpy_helper.from_array(np.ones(3, dtype=np.float32), name='inn_sc')
inn_b  = numpy_helper.from_array(np.zeros(3, dtype=np.float32), name='inn_b')

# Small MatMul weight
np.random.seed(7)
mm_w = numpy_helper.from_array(
    np.random.randn(1, 1, 8, 8).astype(np.float32) * 0.2, name='mm_w')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 8, 8])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

nodes = [
    helper.make_node('Mul', ['X', 'zeros'], ['mze_out']),
    helper.make_node('Add', ['mze_out', 'scale_const'], ['add_out']),
    helper.make_node('InstanceNormalization',
                     ['add_out', 'inn_sc', 'inn_b'], ['inn_out'], epsilon=1e-5),
    helper.make_node('MatMul', ['inn_out', 'mm_w'], ['Y']),
]

graph = helper.make_graph(nodes, 'g', [X], [Y],
    initializer=[zeros, scale_const, inn_sc, inn_b, mm_w])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
model = shape_inference.infer_shapes(model)

INPUT = np.random.randn(1, 3, 8, 8).astype(np.float32)

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
