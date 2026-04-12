"""
Bug v2-0020 — ORT: LayerNorm (axis=0, eps=0) + Resize(nearest, half_pixel) wrong under ORT_ENABLE_ALL
Compiler  : OnnxRuntime
Root cause: Adaptive LayerNorm + residual-add path followed by Resize(nearest,
            round_prefer_floor) triggers incorrect fusion / constant-folding under
            ORT_ENABLE_ALL graph optimizations when the LayerNorm axis spans the
            full tensor and epsilon is exactly zero.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

# Model: Input(1,64,16,16)
# -> LayerNorm(axis=0, eps=0)
# -> LayerNorm(axis=0, eps=0)  [residual-add path]
# -> Resize(nearest, round_prefer_floor, half_pixel, scales=[1,1,2,2])

ln_scale1 = np.ones(16, dtype=np.float32)
ln_bias1  = np.zeros(16, dtype=np.float32)
asc       = np.ones((1, 1, 1, 16), dtype=np.float32)
ab        = np.zeros((1, 1, 1, 16), dtype=np.float32)
ln_scale2 = np.ones(16, dtype=np.float32)
ln_bias2  = np.zeros(16, dtype=np.float32)
residual  = rng.standard_normal((1, 64, 16, 16)).astype(np.float32) * 0.01
roi       = np.array([], dtype=np.float32)
scales    = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

nodes = [
    helper.make_node('LayerNormalization', ['X', 'ln_sc1', 'ln_b1'], ['ln1_out'],
                     axis=-1, epsilon=0.0),
    helper.make_node('Mul', ['ln1_out', 'asc'], ['sc_out']),
    helper.make_node('Add', ['sc_out', 'ab'], ['ad_out']),
    helper.make_node('LayerNormalization', ['ad_out', 'ln_sc2', 'ln_b2'], ['ln2_out'],
                     axis=-1, epsilon=0.0),
    helper.make_node('Add', ['ln2_out', 'res'], ['res_out']),
    helper.make_node('Resize', ['res_out', 'roi', 'scales'], ['Y'],
                     coordinate_transformation_mode='half_pixel',
                     mode='nearest',
                     nearest_mode='round_prefer_floor'),
]

initializers = [
    numpy_helper.from_array(ln_scale1, 'ln_sc1'),
    numpy_helper.from_array(ln_bias1,  'ln_b1'),
    numpy_helper.from_array(asc,       'asc'),
    numpy_helper.from_array(ab,        'ab'),
    numpy_helper.from_array(ln_scale2, 'ln_sc2'),
    numpy_helper.from_array(ln_bias2,  'ln_b2'),
    numpy_helper.from_array(residual,  'res'),
    numpy_helper.from_array(roi,       'roi'),
    numpy_helper.from_array(scales,    'scales'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0020',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64, 16, 16])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64, 32, 32])],
    initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8

INPUT = rng.standard_normal((1, 64, 16, 16)).astype(np.float32)

# Optimized session (ORT_ENABLE_ALL)
sess_opt = ort.InferenceSession(model.SerializeToString(),
                                providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

# Reference session (ORT_DISABLE_ALL)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model.SerializeToString(), sess_options=opts,
                                providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

diff = float(np.max(np.abs(got - expected)))
print(f'max_diff={diff:.4e}')
print(f'PASS={diff < 1e-4}')
