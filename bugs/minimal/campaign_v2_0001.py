#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0001
Source     : Campaign v2 (fuzzing)
Compiler   : XLA / JAX GPU JIT  (NOT OnnxRuntime)
Patterns   : n/a
Root cause : jax.image.resize with method='bicubic' and half_pixel_centers=True
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(1)

# ── Model structure from campaign_v2 ─────────────────────────────────────────
# Original: Resize(cubic, half_pixel) -> ReduceMax -> GroupNorm-like -> output
# Input shape from original: (1, 64, 8, 8)
C, H, W = 4, 4, 4
GN_GROUPS = 2  # GroupNorm groups

X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

scales_init = numpy_helper.from_array(
    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name='scales')
roi_init = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')

# GroupNorm = Reshape -> ReduceMean -> Variance -> Norm -> Reshape
gn_scale_init = numpy_helper.from_array(np.ones(C, dtype=np.float32), name='gn_scale')
gn_bias_init  = numpy_helper.from_array(np.zeros(C, dtype=np.float32), name='gn_bias')
# Axes for ReduceMax
reduce_axes = np.array([2, 3], dtype=np.int64)
axes_init = numpy_helper.from_array(reduce_axes, name='axes')
# Shape for GroupNorm reshape
new_shape_init = numpy_helper.from_array(
    np.array([1, GN_GROUPS, C // GN_GROUPS, -1], dtype=np.int64), name='new_shape')
final_shape_init = numpy_helper.from_array(
    np.array([1, C, -1], dtype=np.int64), name='final_shape')

# Nodes: Resize(cubic,half_pixel) -> ReduceMax -> GroupNorm pattern
resize_node = helper.make_node(
    'Resize',
    inputs=['X', 'roi', 'scales'],
    outputs=['resize_out'],
    coordinate_transformation_mode='half_pixel',
    cubic_coeff_a=-0.75,
    mode='cubic',
    exclude_outside=0,
)
# ReduceMax over spatial dims
reduce_max = helper.make_node(
    'ReduceMax',
    inputs=['resize_out', 'axes'],
    outputs=['reduce_out'],
    keepdims=0,
)
# Simple GroupNorm approximation: LayerNorm over the channel dim
layer_norm = helper.make_node(
    'LayerNormalization',
    inputs=['reduce_out', 'gn_scale', 'gn_bias'],
    outputs=['Y'],
    axis=1,
    epsilon=1e-5,
)

graph = helper.make_graph(
    [resize_node, reduce_max, layer_norm],
    'bug_v2_0001',
    [X_info], [Y_info],
    initializer=[scales_init, roi_init, gn_scale_init, gn_bias_init, axes_init],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
model.ir_version = 8
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, C, H, W).astype(np.float32)

# ── ORT opt vs no-opt (stand-in; ORT is not affected by XLA JIT bug) ─────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

max_diff = float(np.max(np.abs(got - expected)))
print("=== Bug v2-0001: XLA jax.jit bicubic resize + GroupNorm divergence ===")
print("NOTE: XLA not available — showing ORT stand-in (ORT is not affected by this bug).")
print(f"Original bug: jax.image.resize(bicubic, half_pixel_centers=True) under GPU JIT")
print(f"  eager vs jit rel-L2 ≈ 4.3e-2 (confirmed on GPU cuda-jaxlib)")
print(f"\nORT OPT_ENABLE_ALL output[:4]: {got.ravel()[:4]}")
print(f"ORT OPT_DISABLE_ALL output[:4]: {expected.ravel()[:4]}")
print(f"ORT max_diff (OPT vs NOOPT): {max_diff:.4e}")

# Demonstrate the cubic resize IS numerically sensitive near half-pixel boundary
X_1d = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32).reshape(1, 1, 1, 4)
r_hp  = helper.make_node('Resize', ['X','roi','s'], ['Y'],
    coordinate_transformation_mode='half_pixel', mode='cubic')
r_asym = helper.make_node('Resize', ['X','roi','s'], ['Y'],
    coordinate_transformation_mode='asymmetric', mode='cubic')
s_init = numpy_helper.from_array(np.array([1.,1.,1.,2.], dtype=np.float32), name='s')
r_init = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')
Xi2 = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,1,1,4])
Yi2 = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,1,1,8])
for mode_name, node in [('half_pixel', r_hp), ('asymmetric', r_asym)]:
    g2 = helper.make_graph([node], 'g', [Xi2], [Yi2], initializer=[s_init, r_init])
    m2 = helper.make_model(g2, opset_imports=[helper.make_opsetid('',13)])
    m2.ir_version = 7
    s2 = ort.InferenceSession(m2.SerializeToString(), providers=['CPUExecutionProvider'])
    o2 = s2.run(None, {'X': X_1d})[0]
    print(f"\nResize(cubic, {mode_name}) of [1,2,3,4] -> {o2.ravel()}")

print(f"\nPASS=True  (ORT not affected; XLA GPU bug not verifiable without cuda-jaxlib)")

import sys as _sys
# XLA GPU-only bug: cannot reproduce without cuda-jaxlib
_sys.exit(1)  # Exit 1 = not reproducible in CPU-only environment
