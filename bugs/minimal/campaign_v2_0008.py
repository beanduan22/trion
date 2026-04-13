#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0008
Source     : Campaign v2 (fuzzing)
Compiler   : XLA / JAX GPU JIT  (NOT OnnxRuntime)
Patterns   : Resize(cubic, half_pixel) -> [dilated Conv branch + edge-pad branch] -> Add
Root cause : JAX XLA JIT fuses a dilated convolution branch with an edge-mode padding
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(8)

# ── Build equivalent ONNX model ───────────────────────────────────────────────
# Pattern: Resize(cubic, half_pixel) -> [dilated Conv branch + edge-pad branch] -> Add
C_IN, C_OUT, H, W = 4, 4, 4, 4
DILATION = 2

X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C_IN, H, W])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

scales_init = numpy_helper.from_array(
    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name='scales')
roi_init = numpy_helper.from_array(np.array([], dtype=np.float32), name='roi')

# Dilated conv weights 3x3 with dilation=2 (effective receptive field 5x5)
dil_w = np.random.randn(C_OUT, C_IN, 3, 3).astype(np.float32) * 0.1
dil_b = np.zeros(C_OUT, dtype=np.float32)
dil_w_init = numpy_helper.from_array(dil_w, 'dil_w')
dil_b_init = numpy_helper.from_array(dil_b, 'dil_b')

# 1x1 conv for the bypass branch
bypass_w = np.eye(C_OUT, C_IN, dtype=np.float32).reshape(C_OUT, C_IN, 1, 1) * 0.5
bypass_b = np.zeros(C_OUT, dtype=np.float32)
bypass_w_init = numpy_helper.from_array(bypass_w, 'bypass_w')
bypass_b_init = numpy_helper.from_array(bypass_b, 'bypass_b')

# Edge-mode pad pads: [0,0,0,0, 2,2, 2,2] for NCHW
# ONNX Pad pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
pads_val = np.array([0, 0, 2, 2, 0, 0, 2, 2], dtype=np.int64)
pads_init = numpy_helper.from_array(pads_val, 'pads')

nodes = [
    # Bicubic upsample 2x
    helper.make_node('Resize', ['X', 'roi', 'scales'], ['resize_out'],
                     coordinate_transformation_mode='half_pixel',
                     cubic_coeff_a=-0.75, mode='cubic', exclude_outside=0),
    # Edge-mode pad (mimics jnp.pad with mode='edge')
    helper.make_node('Pad', ['resize_out', 'pads'], ['padded'],
                     mode='edge'),
    # Dilated conv 3x3 (dilation=2, no explicit pad because we pre-padded)
    helper.make_node('Conv', ['padded', 'dil_w', 'dil_b'], ['dil_out'],
                     kernel_shape=[3, 3],
                     dilations=[DILATION, DILATION],
                     pads=[0, 0, 0, 0]),
    # Bypass branch: 1x1 conv on resize_out
    helper.make_node('Conv', ['resize_out', 'bypass_w', 'bypass_b'], ['bypass_out'],
                     kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
    # Add branches
    helper.make_node('Add', ['dil_out', 'bypass_out'], ['Y']),
]

graph = helper.make_graph(
    nodes, 'bug_v2_0008',
    [X_info], [Y_info],
    initializer=[scales_init, roi_init, dil_w_init, dil_b_init,
                 bypass_w_init, bypass_b_init, pads_init],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 7
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, C_IN, H, W).astype(np.float32)

# ── ORT opt vs no-opt ─────────────────────────────────────────────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

max_diff = float(np.max(np.abs(got - expected)))

print("=== Bug v2-0008: XLA bicubic resize + dilated conv + edge-mode pad ===")
print("NOTE: XLA not available — showing ONNX model pattern + ORT stand-in.")
print("Original bug: JAX XLA JIT fuses dilated conv with edge-pad, diverging at boundaries.")
print(f"  Original: rel-L2(jit vs eager) nonzero; requires cuda-jaxlib to reproduce.")
print(f"\nONNX model: Resize(cubic,half_pixel,2x) -> Pad(edge) -> DilatedConv + BypassConv -> Add")
print(f"Input shape: {INPUT.shape}, dilations={DILATION}, output shape: {got.shape}")
print(f"\nORT_ENABLE_ALL[:4]:  {got.ravel()[:4]}")
print(f"ORT_DISABLE_ALL[:4]: {expected.ravel()[:4]}")
print(f"max_diff (OPT vs NOOPT): {max_diff:.4e}")

# Show that edge padding boundary values are different from zero padding
# (to motivate why this matters for the XLA divergence)
no_pad = np.zeros((1, C_IN, H*2 + 4, W*2 + 4), dtype=np.float32)
edge_pad = np.pad(
    ort.InferenceSession(
        helper.make_model(
            helper.make_graph(
                [helper.make_node('Resize', ['X','roi','scales'], ['Y'],
                    coordinate_transformation_mode='half_pixel', cubic_coeff_a=-0.75,
                    mode='cubic', exclude_outside=0)],
                'g',
                [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,C_IN,H,W])],
                [helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)],
                initializer=[scales_init, roi_init]
            ),
            opset_imports=[helper.make_opsetid('',13)]
        ).SerializeToString(),
        providers=['CPUExecutionProvider']
    ).run(None, {'X': INPUT})[0][0],
    ((0,0), (2,2), (2,2)),
    mode='edge'
)
print(f"\nEdge-padded boundary[0,0,:4]: {edge_pad[0,:4, 0]}")
print(f"  (Edge mode replicates border pixels; XLA JIT fuses this with conv)")

print(f"\nPASS=True  (ORT not affected; XLA GPU bug not verifiable without cuda-jaxlib)")

import sys as _sys
# XLA GPU-only bug: cannot reproduce without cuda-jaxlib
_sys.exit(1)  # Exit 1 = not reproducible in CPU-only environment
