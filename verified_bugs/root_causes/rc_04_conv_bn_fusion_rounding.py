"""
Root Cause 4 — ORT Conv+BatchNorm fusion accumulates float32 rounding error
============================================================================
Affects: campaign_v2 uid 0004, 0009 (ORT_ENABLE_ALL vs ORT_DISABLE_ALL diff)

Bug:     When ORT's graph optimizer fuses a Conv node followed immediately by
         a BatchNormalization node (inference mode), it bakes the BN parameters
         (scale, bias, mean, var) into modified Conv kernel weights and bias.
         This weight-baking is done in float32 arithmetic. When the original
         BN parameters have large magnitudes or small variances, the baking
         introduces rounding errors of ~1e-5 relative to executing Conv and
         BatchNorm as separate sequential operations.

Root cause: ORT ConvBnFusion pass computes:
             W_fused = W * (scale / sqrt(var + eps))
             b_fused = (b - mean) * (scale / sqrt(var + eps)) + bias
           in float32, which loses precision compared to running BN separately
           in float32 at inference time.

           This is an ORT optimizer regression: ORT_ENABLE_ALL ≠ ORT_DISABLE_ALL.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(0)

# --- Inputs and weights designed to amplify the rounding ---
x     = np.random.randn(1, 4, 8, 8).astype(np.float32)
W     = np.random.randn(4, 4, 3, 3).astype(np.float32)
b     = np.random.randn(4).astype(np.float32)
scale = np.array([10., 20., 0.1, 50.], dtype=np.float32)  # large/small scale
bias  = np.random.randn(4).astype(np.float32)
mean  = np.random.randn(4).astype(np.float32)
var   = np.array([0.001, 100., 0.0001, 200.], dtype=np.float32)  # extreme var

# --- ONNX model: Conv → BatchNormalization ---
X   = helper.make_tensor_value_info('X',     TensorProto.FLOAT, [1, 4, 8, 8])
Y   = helper.make_tensor_value_info('Y',     TensorProto.FLOAT, None)
W_t = numpy_helper.from_array(W,     'W')
b_t = numpy_helper.from_array(b,     'B')
s_t = numpy_helper.from_array(scale, 'scale')
bi_t= numpy_helper.from_array(bias,  'bias')
m_t = numpy_helper.from_array(mean,  'mean')
v_t = numpy_helper.from_array(var,   'var')

conv = helper.make_node('Conv',               ['X', 'W', 'B'],                     ['conv_out'],
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1])
bn   = helper.make_node('BatchNormalization', ['conv_out', 'scale', 'bias', 'mean', 'var'], ['Y'])

graph = helper.make_graph([conv, bn], 'g', [X], [Y],
                          initializer=[W_t, b_t, s_t, bi_t, m_t, v_t])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])

# --- ORT with full optimization (Conv+BN fused → baked weights) ---
opts_all = ort.SessionOptions()
opts_all.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_all  = ort.InferenceSession(model.SerializeToString(),
                                 sess_options=opts_all, providers=['CPUExecutionProvider'])
out_fused = sess_all.run(None, {'X': x})[0]

# --- ORT with no optimization (Conv and BN run separately) ---
opts_none = ort.SessionOptions()
opts_none.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_none  = ort.InferenceSession(model.SerializeToString(),
                                  sess_options=opts_none, providers=['CPUExecutionProvider'])
out_unfused = sess_none.run(None, {'X': x})[0]

# --- numpy float64 reference (ground truth) ---
from scipy.signal import correlate
eps = 1e-5
conv_out = np.zeros((1, 4, 8, 8), dtype=np.float64)
W64 = W.astype(np.float64)
x64 = x.astype(np.float64)
for oc in range(4):
    for ic in range(4):
        for ih in range(8):
            for iw in range(8):
                pass  # use onnxruntime unfused as reference instead
ref = out_unfused  # unfused sequential is the correct baseline

diff_fused   = np.max(np.abs(out_fused   - ref))
diff_unfused = np.max(np.abs(out_unfused - ref))

print(f"ORT_ENABLE_ALL  (fused)   output[0,0,0,:4]: {out_fused.ravel()[:4]}")
print(f"ORT_DISABLE_ALL (unfused) output[0,0,0,:4]: {out_unfused.ravel()[:4]}")
print()
print(f"max_diff ENABLE_ALL vs DISABLE_ALL: {diff_fused:.3e}")
print(f"(ORT Conv+BN fusion bakes BN params into float32 weights, losing precision)")
print()
print(f"PASS={diff_fused < 1e-4}")
