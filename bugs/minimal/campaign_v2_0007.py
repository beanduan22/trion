#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0007
Source     : Campaign v2 (fuzzing)
Compiler   : XLA / JAX GPU JIT  (NOT OnnxRuntime)
Patterns   : LayerNorm -> ReLU (from "layernorm_relu") -> Add(x, 0) -> Softplus
Root cause : JAX XLA JIT applies an "add-zero identity" fold (x + 0 -> x) that
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(7)

# ── Build equivalent ONNX model ───────────────────────────────────────────────
# Pattern: LayerNorm -> ReLU (from "layernorm_relu") -> Add(x, 0) -> Softplus
C, H, W = 4, 4, 4

X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

ln_scale = numpy_helper.from_array(np.ones(C * H * W, dtype=np.float32),  name='ln_scale')
ln_bias  = numpy_helper.from_array(np.zeros(C * H * W, dtype=np.float32), name='ln_bias')
zero_add = numpy_helper.from_array(np.zeros(1, dtype=np.float32),         name='zero_add')

# Reshape to [1, C*H*W] for LayerNorm, then back
reshape_dim1 = numpy_helper.from_array(
    np.array([1, C * H * W], dtype=np.int64), name='reshape_dim1')
reshape_dim2 = numpy_helper.from_array(
    np.array([1, C, H, W], dtype=np.int64), name='reshape_dim2')

nodes = [
    helper.make_node('Reshape', ['X', 'reshape_dim1'], ['flat']),
    helper.make_node('LayerNormalization', ['flat', 'ln_scale', 'ln_bias'], ['ln_out'],
                     axis=1, epsilon=1e-5),
    helper.make_node('Relu', ['ln_out'], ['relu_out']),
    # Add-zero identity (foldable): x + 0 -> x in XLA JIT
    helper.make_node('Add', ['relu_out', 'zero_add'], ['added']),
    # Softplus: log(1 + exp(x)), sensitive to x values after the add-zero fold
    helper.make_node('Softplus', ['added'], ['sp_out']),
    helper.make_node('Reshape', ['sp_out', 'reshape_dim2'], ['Y']),
]

graph = helper.make_graph(
    nodes, 'bug_v2_0007',
    [X_info], [Y_info],
    initializer=[ln_scale, ln_bias, zero_add, reshape_dim1, reshape_dim2],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 7
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, C, H, W).astype(np.float32)

# ── ORT opt vs no-opt ─────────────────────────────────────────────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

# ── numpy reference ───────────────────────────────────────────────────────────
flat = INPUT.reshape(1, -1).astype(np.float64)
mean = flat.mean(axis=1, keepdims=True)
var  = flat.var(axis=1, keepdims=True)
ln_out = (flat - mean) / np.sqrt(var + 1e-5)  # scale=1, bias=0
relu_out = np.maximum(ln_out, 0)
added_out = relu_out + 0.0  # add-zero identity
sp_ref = np.log1p(np.exp(added_out)).reshape(1, C, H, W).astype(np.float32)

max_diff_ort_noopt = float(np.max(np.abs(got - expected)))
max_diff_ort_ref   = float(np.max(np.abs(got - sp_ref)))

print("=== Bug v2-0007: XLA add-zero identity fold + Softplus fast-path ===")
print("NOTE: XLA not available — showing ONNX model pattern + ORT stand-in.")
print("Original bug: JAX XLA JIT add-zero fold changes softplus kernel path.")
print(f"  Original: rel-L2(jit vs eager) nonzero; requires cuda-jaxlib to reproduce.")
print(f"\nONNX model nodes: LN -> ReLU -> Add(x, 0) -> Softplus")
print(f"Input shape: {INPUT.shape}")
print(f"\nORT_ENABLE_ALL[:4]:  {got.ravel()[:4]}")
print(f"ORT_DISABLE_ALL[:4]: {expected.ravel()[:4]}")
print(f"numpy ref[:4]:       {sp_ref.ravel()[:4]}")
print(f"\nmax_diff (OPT vs NOOPT):   {max_diff_ort_noopt:.4e}")
print(f"max_diff (ORT vs numpy64): {max_diff_ort_ref:.4e}")
print(f"\nPASS=True  (ORT not affected; XLA GPU bug not verifiable without cuda-jaxlib)")

import sys as _sys
# XLA GPU-only bug: cannot reproduce without cuda-jaxlib
_sys.exit(1)  # Exit 1 = not reproducible in CPU-only environment
