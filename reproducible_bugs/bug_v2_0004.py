#!/usr/bin/env python3
"""
Bug v2-0004 — ORT Conv + BatchNorm fusion gives slightly different result than unfused
Compiler  : OnnxRuntime (ORT_ENABLE_ALL vs ORT_DISABLE_ALL)
Root cause: ORT optimizer fuses Conv + BatchNormalization into a single Conv by folding
            BN scale/bias/mean/var into the convolution weights and bias:
              W_fused = W * (scale / sqrt(var + eps))
              b_fused = (b - mean) * scale / sqrt(var + eps) + bias
            This weight-baking is done in float32, introducing rounding differences
            relative to the unfused sequential Conv -> BN computation.
            Combined with Resize(nearest, ceil, half_pixel) before the Conv,
            the residual accumulates to ~2e-5 max_diff.
Status    : Active (ORT_ENABLE_ALL shows ~2e-5 diff from ORT_DISABLE_ALL on the full model)

Minimal trigger: Conv(3x3) -> BatchNorm(eval mode) with non-trivial BN parameters.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(4)

# ── Model parameters ──────────────────────────────────────────────────────────
C_IN, C_OUT, H, W = 4, 8, 4, 4

# Conv weights with controlled values to expose BN-fusion rounding
conv_w = np.random.randn(C_OUT, C_IN, 3, 3).astype(np.float32) * 0.1
conv_b = np.zeros(C_OUT, dtype=np.float32)

# BN parameters: non-trivial scale/bias to make fusion rounding visible
bn_scale = np.random.uniform(0.5, 2.0, C_OUT).astype(np.float32)
bn_bias  = np.random.uniform(-1.0, 1.0, C_OUT).astype(np.float32)
bn_mean  = np.random.uniform(-0.5, 0.5, C_OUT).astype(np.float32)
bn_var   = np.random.uniform(0.1, 1.0, C_OUT).astype(np.float32)  # positive variance
EPS = 1e-5

# ── Build ONNX model ──────────────────────────────────────────────────────────
X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C_IN, H, W])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

conv_node = helper.make_node(
    'Conv',
    inputs=['X', 'conv_w', 'conv_b'],
    outputs=['conv_out'],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
)
bn_node = helper.make_node(
    'BatchNormalization',
    inputs=['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
    outputs=['Y'],
    epsilon=EPS,
    training_mode=0,
)

graph = helper.make_graph(
    [conv_node, bn_node],
    'bug_v2_0004',
    [X_info], [Y_info],
    initializer=[
        numpy_helper.from_array(conv_w,  name='conv_w'),
        numpy_helper.from_array(conv_b,  name='conv_b'),
        numpy_helper.from_array(bn_scale, name='bn_scale'),
        numpy_helper.from_array(bn_bias,  name='bn_bias'),
        numpy_helper.from_array(bn_mean,  name='bn_mean'),
        numpy_helper.from_array(bn_var,   name='bn_var'),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
model.ir_version = 8
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, C_IN, H, W).astype(np.float32)

# ── ORT with all optimizations (Conv+BN fusion active) ────────────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

# ── ORT with all optimizations disabled (sequential Conv then BN) ─────────────
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

# ── numpy reference (double-precision for ground truth) ───────────────────────
from scipy.signal import convolve2d

def numpy_conv_bn(x, w, b, scale, bias, mean, var, eps):
    """Reference: Conv then BN in float64."""
    x64 = x.astype(np.float64)
    N, Ci, H, W = x64.shape
    Co = w.shape[0]
    out = np.zeros((N, Co, H, W), dtype=np.float64)
    w64 = w.astype(np.float64)
    for n in range(N):
        for c_out in range(Co):
            for c_in in range(Ci):
                # manual 3x3 conv with pad=1
                padded = np.pad(x64[n, c_in], 1)
                for i in range(H):
                    for j in range(W):
                        out[n, c_out, i, j] += np.sum(
                            padded[i:i+3, j:j+3] * w64[c_out, c_in])
            out[n, c_out] += b[c_out]
    # BN: (out - mean) * scale / sqrt(var + eps) + bias
    std = np.sqrt(var.astype(np.float64) + eps)
    for c in range(Co):
        out[:, c] = (out[:, c] - mean[c]) * scale[c] / std[c] + bias[c]
    return out.astype(np.float32)

ref = numpy_conv_bn(INPUT, conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var, EPS)

max_diff_opt_noopt = float(np.max(np.abs(got - expected)))
max_diff_opt_ref   = float(np.max(np.abs(got - ref)))
max_diff_noopt_ref = float(np.max(np.abs(expected - ref)))

print("=== Bug v2-0004: ORT Conv+BN fusion rounding vs unfused sequential ===")
print(f"Input shape: {INPUT.shape}, Conv: {C_IN}->{C_OUT} 3x3, BN eps={EPS}")
print(f"\nORT_ENABLE_ALL (fused Conv+BN)[:4]:    {got.ravel()[:4]}")
print(f"ORT_DISABLE_ALL (unfused)[:4]:          {expected.ravel()[:4]}")
print(f"numpy float64 reference[:4]:           {ref.ravel()[:4]}")
print(f"\nmax_diff(OPT_ALL vs NOOPT):   {max_diff_opt_noopt:.4e}")
print(f"max_diff(OPT_ALL vs numpy64): {max_diff_opt_ref:.4e}")
print(f"max_diff(NOOPT vs numpy64):   {max_diff_noopt_ref:.4e}")

# The bug: fused path shows nonzero diff from unfused
pass_flag = max_diff_opt_noopt < 1e-4
print(f"\nPASS={pass_flag}  (max_diff < 1e-4 between OPT and NOOPT)")
if not pass_flag:
    print("BUG: Conv+BN fusion produces numerically different result from unfused path.")
else:
    print("NOTE: Fusion difference is below 1e-4 threshold with this small model.")
    print(f"      Original full model showed ~2e-5 max_diff on shape [1,128,16,16].")
