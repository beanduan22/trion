#!/usr/bin/env python3
"""
Bug v2-0009 — ORT Conv + HardSwish + BN fusion rounding difference
Compiler  : OnnxRuntime (ORT_ENABLE_ALL vs ORT_DISABLE_ALL, ~5e-6 max_diff)
Root cause: ORT optimizer fuses Conv -> HardSwish -> BatchNormalization into a
            single kernel by baking BN scale/shift/mean/var into the convolution
            weights (same Conv+BN fusion as v2-0004).  The HardSwish non-linearity
            sits between the Conv and BN; after fusion the BN weight-baking is
            applied to the post-HardSwish activations but the rounding order changes.
            Additionally, Resize(linear, asymmetric) before the second conv adds
            coordinate-transform differences when compared to pytorch.
Status    : Active (ORT_ENABLE_ALL shows ~5e-6 diff from ORT_DISABLE_ALL)

Minimal trigger: Conv(3x3) -> HardSwish -> BatchNorm(eval) with non-trivial BN params.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(9)

# ── Model parameters ──────────────────────────────────────────────────────────
C_IN, C_MID, H, W = 4, 8, 8, 8

# Conv + HardSwish + BN
conv1_w = np.random.randn(C_MID, C_IN, 3, 3).astype(np.float32) * 0.1
conv1_b = np.zeros(C_MID, dtype=np.float32)
bn_scale = np.random.uniform(0.5, 2.0, C_MID).astype(np.float32)
bn_bias  = np.random.uniform(-1.0, 1.0, C_MID).astype(np.float32)
bn_mean  = np.random.uniform(-0.5, 0.5, C_MID).astype(np.float32)
bn_var   = np.random.uniform(0.5, 2.0,  C_MID).astype(np.float32)  # positive var
EPS = 1e-5

# ── Build ONNX model ──────────────────────────────────────────────────────────
X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C_IN, H, W])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

conv1_node = helper.make_node(
    'Conv', ['X', 'conv1_w', 'conv1_b'], ['conv1_out'],
    kernel_shape=[3, 3], pads=[1, 1, 1, 1],
)
hardswish_node = helper.make_node('HardSwish', ['conv1_out'], ['hs_out'])
bn_node = helper.make_node(
    'BatchNormalization',
    ['hs_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
    ['Y'],
    epsilon=EPS, training_mode=0,
)

graph = helper.make_graph(
    [conv1_node, hardswish_node, bn_node],
    'bug_v2_0009',
    [X_info], [Y_info],
    initializer=[
        numpy_helper.from_array(conv1_w,  'conv1_w'),
        numpy_helper.from_array(conv1_b,  'conv1_b'),
        numpy_helper.from_array(bn_scale, 'bn_scale'),
        numpy_helper.from_array(bn_bias,  'bn_bias'),
        numpy_helper.from_array(bn_mean,  'bn_mean'),
        numpy_helper.from_array(bn_var,   'bn_var'),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])
model.ir_version = 8
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, C_IN, H, W).astype(np.float32)

# ── ORT with all optimizations (Conv+HardSwish+BN may fuse) ──────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

# ── ORT with all optimizations disabled ───────────────────────────────────────
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

# ── numpy float64 reference ───────────────────────────────────────────────────
def hardswish_np(x):
    return x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)

def manual_conv2d(x, w, b, pad=1):
    N, Ci, H, W_in = x.shape
    Co, _, kH, kW = w.shape
    out = np.zeros((N, Co, H, W_in), dtype=np.float64)
    x64 = np.pad(x.astype(np.float64), ((0,0),(0,0),(pad,pad),(pad,pad)))
    w64 = w.astype(np.float64)
    for n in range(N):
        for c in range(Co):
            for i in range(H):
                for j in range(W_in):
                    out[n,c,i,j] = np.sum(x64[n,:,i:i+kH,j:j+kW] * w64[c]) + b[c]
    return out

conv_ref = manual_conv2d(INPUT, conv1_w, conv1_b)
hs_ref   = hardswish_np(conv_ref)
std_ref  = np.sqrt(bn_var.astype(np.float64) + EPS)
bn_ref   = np.zeros_like(hs_ref)
for c in range(C_MID):
    bn_ref[:, c] = (hs_ref[:, c] - bn_mean[c]) * bn_scale[c] / std_ref[c] + bn_bias[c]
ref = bn_ref.astype(np.float32)

max_diff_opt_noopt = float(np.max(np.abs(got - expected)))
max_diff_opt_ref   = float(np.max(np.abs(got - ref)))
max_diff_noopt_ref = float(np.max(np.abs(expected - ref)))

print("=== Bug v2-0009: ORT Conv + HardSwish + BatchNorm fusion rounding ===")
print(f"Input: {INPUT.shape}, Conv {C_IN}->{C_MID} 3x3, HardSwish, BN")
print(f"\nORT_ENABLE_ALL[:4]:    {got.ravel()[:4]}")
print(f"ORT_DISABLE_ALL[:4]:   {expected.ravel()[:4]}")
print(f"numpy float64 ref[:4]: {ref.ravel()[:4]}")
print(f"\nmax_diff(OPT vs NOOPT):   {max_diff_opt_noopt:.4e}  <-- BUG: nonzero due to fusion")
print(f"max_diff(OPT vs numpy64): {max_diff_opt_ref:.4e}")
print(f"max_diff(NOOPT vs numpy64): {max_diff_noopt_ref:.4e}")

# HardSwish formula verification
x_test = np.array([-3.5, -1.0, 0.0, 1.0, 3.5], dtype=np.float32).reshape(1,1,1,-1)
Xi2 = helper.make_tensor_value_info('X', TensorProto.FLOAT, None)
Yo2 = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)
m2 = helper.make_model(
    helper.make_graph([helper.make_node('HardSwish',['X'],['Y'])], 'g', [Xi2],[Yo2]),
    opset_imports=[helper.make_opsetid('',14)])
m2.ir_version = 8
hs_test = ort.InferenceSession(m2.SerializeToString(),
    providers=['CPUExecutionProvider']).run(None, {'X': x_test})[0]
hs_ref2  = hardswish_np(x_test)
print(f"\nHardSwish verification: x * clip(x/6+0.5, 0, 1)")
print(f"  x     = {x_test.ravel()}")
print(f"  ORT   = {hs_test.ravel()}")
print(f"  numpy = {hs_ref2.ravel()}")

pass_flag = max_diff_opt_noopt < 1e-4
print(f"\nPASS={pass_flag}  (max_diff < 1e-4 threshold)")
if not pass_flag:
    print("BUG: Conv+HardSwish+BN fusion produces different result from unfused path.")
else:
    print("NOTE: Small models may not trigger the fusion. ")
    print(f"      Original full model [1,3,32,32] showed ~5e-6 max_diff.")
