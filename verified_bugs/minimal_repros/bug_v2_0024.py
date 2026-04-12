"""
Bug v2-0024 — ORT: two Resize nodes merged wrongly (linear half_pixel + nearest_ceil)
Compiler  : OnnxRuntime
Root cause: ORT's graph optimizer incorrectly merges / reorders two consecutive Resize operations
            when the first uses half_pixel+linear and the second uses half_pixel+nearest_ceil.
            The shared MatMul weight matrices between the two resize nodes create an aliasing
            situation that the optimizer exploits incorrectly.
Pattern   : MM4D -> Resize(linear, half_pixel, 2x) -> MM4D -> MM4D ->
            Resize(nearest_ceil, half_pixel, 2x)
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

rng = np.random.default_rng(42)

C, H, W = 3, 32, 32

mm_w1 = rng.standard_normal((1, 1, H, W)).astype(np.float32) * 0.3
roi   = np.array([], dtype=np.float32)
sc1   = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
# after resize1: (1,3,64,64)
mm_w2 = rng.standard_normal((1, 1, 64, 64)).astype(np.float32) * 0.2
# shared weight for second matmul
mm_w3 = mm_w2.copy()
sc2   = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

nodes = [
    helper.make_node('MatMul', ['X', 'mm_w1'], ['mm1_out']),
    helper.make_node('Resize', ['mm1_out', 'roi', 'sc1'], ['r1_out'],
                     coordinate_transformation_mode='half_pixel',
                     mode='linear'),
    helper.make_node('MatMul', ['r1_out', 'mm_w2'], ['mm2_out']),
    helper.make_node('MatMul', ['mm2_out', 'mm_w3'], ['mm3_out']),
    helper.make_node('Resize', ['mm3_out', 'roi', 'sc2'], ['Y'],
                     coordinate_transformation_mode='half_pixel',
                     mode='nearest',
                     nearest_mode='ceil'),
]

initializers = [
    numpy_helper.from_array(mm_w1, 'mm_w1'),
    numpy_helper.from_array(roi,   'roi'),
    numpy_helper.from_array(sc1,   'sc1'),
    numpy_helper.from_array(mm_w2, 'mm_w2'),
    numpy_helper.from_array(mm_w3, 'mm_w3'),
    numpy_helper.from_array(sc2,   'sc2'),
]

graph = helper.make_graph(
    nodes,
    'bug_v2_0024',
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
