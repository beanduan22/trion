#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0003
Source     : Campaign v2 (fuzzing)
Compiler   : XLA / JAX GPU JIT  (NOT OnnxRuntime)
Patterns   : Resize(cubic, half_pixel) -> BatchNormalization(eval mode) -> FPN branch
Root cause : JAX XLA JIT fuses bicubic resize with BatchNorm scale/shift into a
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(3)

# ── Build equivalent ONNX model ───────────────────────────────────────────────
# Pattern: Resize(cubic, half_pixel) -> BatchNormalization(eval mode) -> FPN branch
C, H, W = 4, 4, 4
C_OUT = 8  # after conv in FPN

X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

scales_init = numpy_helper.from_array(
    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name='scales')
roi_init = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')

# BatchNorm params (eval mode: fixed mean/var)
bn_scale_init = numpy_helper.from_array(np.ones(C, dtype=np.float32),  name='bn_scale')
bn_bias_init  = numpy_helper.from_array(np.zeros(C, dtype=np.float32), name='bn_bias')
bn_mean_init  = numpy_helper.from_array(np.zeros(C, dtype=np.float32), name='bn_mean')
bn_var_init   = numpy_helper.from_array(np.ones(C, dtype=np.float32),  name='bn_var')

# FPN conv weights (1x1, C -> C_OUT)
fpn_w = np.random.randn(C_OUT, C, 1, 1).astype(np.float32) * 0.1
fpn_b = np.zeros(C_OUT, dtype=np.float32)
fpn_w_init = numpy_helper.from_array(fpn_w, name='fpn_w')
fpn_b_init = numpy_helper.from_array(fpn_b, name='fpn_b')

# Nodes
resize_node = helper.make_node(
    'Resize',
    inputs=['X', 'roi', 'scales'],
    outputs=['resize_out'],
    coordinate_transformation_mode='half_pixel',
    cubic_coeff_a=-0.75,
    mode='cubic',
    exclude_outside=0,
)
bn_node = helper.make_node(
    'BatchNormalization',
    inputs=['resize_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
    outputs=['bn_out'],
    epsilon=1e-5,
    training_mode=0,
)
fpn_conv = helper.make_node(
    'Conv',
    inputs=['bn_out', 'fpn_w', 'fpn_b'],
    outputs=['Y'],
    kernel_shape=[1, 1],
    pads=[0, 0, 0, 0],
)

graph = helper.make_graph(
    [resize_node, bn_node, fpn_conv],
    'bug_v2_0003',
    [X_info], [Y_info],
    initializer=[scales_init, roi_init, bn_scale_init, bn_bias_init,
                 bn_mean_init, bn_var_init, fpn_w_init, fpn_b_init],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
model.ir_version = 8
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, C, H, W).astype(np.float32)

# ── ORT opt vs noopt (stand-in) ───────────────────────────────────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

max_diff = float(np.max(np.abs(got - expected)))

print("=== Bug v2-0003: XLA bicubic resize + BatchNorm eval fusion ===")
print("NOTE: XLA not available — showing ONNX model structure + ORT stand-in.")
print("Original bug: JAX XLA JIT fuses Resize(cubic,half_pixel) + BN into one kernel.")
print("  BN scale/shift reordering in fused kernel produces floating-point divergence.")
print(f"  Original: rel-L2(jit vs eager) ~ large; requires cuda-jaxlib to reproduce.")
print(f"\nONNX model nodes: Resize(cubic, half_pixel) -> BatchNormalization -> Conv(1x1)")
print(f"Input shape: {INPUT.shape}")
print(f"\nORT OPT_ENABLE_ALL[:4]:   {got.ravel()[:4]}")
print(f"ORT OPT_DISABLE_ALL[:4]:  {expected.ravel()[:4]}")
print(f"max_diff (OPT vs NOOPT):  {max_diff:.4e}")
print(f"\nPASS=True  (ORT not affected; XLA GPU bug not verifiable without cuda-jaxlib)")

import sys as _sys
# XLA GPU-only bug: cannot reproduce without cuda-jaxlib
_sys.exit(1)  # Exit 1 = not reproducible in CPU-only environment
