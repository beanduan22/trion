import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Bug: TF BN folding into Conv applied scale incorrectly in quantization-aware training (issue #43882).
np.random.seed(42)
w      = np.random.randn(8, 4, 3, 3).astype(np.float32)
b_conv = np.random.randn(8).astype(np.float32)
gamma  = np.random.rand(8).astype(np.float32) + 0.5
beta   = np.random.randn(8).astype(np.float32)
r_mean = np.random.randn(8).astype(np.float32)
r_var  = np.random.rand(8).astype(np.float32) + 0.1
x      = np.random.randn(1, 4, 6, 6).astype(np.float32)
eps    = 1e-5

# Reference: unfused Conv + BN
X_ref = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 6, 6])
Y_ref = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
inits_ref = [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b_conv, "Bc"),
             numpy_helper.from_array(gamma, "gamma"), numpy_helper.from_array(beta, "beta"),
             numpy_helper.from_array(r_mean, "rmean"), numpy_helper.from_array(r_var, "rvar")]
c_ref  = helper.make_node("Conv", ["X","W","Bc"], ["conv_out"], pads=[1,1,1,1])
bn_ref = helper.make_node("BatchNormalization", ["conv_out","gamma","beta","rmean","rvar"], ["Y"], epsilon=eps)
g_ref  = helper.make_graph([c_ref, bn_ref], "g", [X_ref], [Y_ref], initializer=inits_ref)
m_ref  = helper.make_model(g_ref, opset_imports=[helper.make_opsetid("", 9)])
ref_out = ort.InferenceSession(m_ref.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

# Correct BN folding math
std    = np.sqrt(r_var + eps)
scale  = gamma / std
w_fold = w * scale[:, None, None, None]
b_fold = (b_conv - r_mean) * scale + beta

# Folded: single Conv with merged BN weights
X_f = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 6, 6])
Y_f = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
inits_f = [numpy_helper.from_array(w_fold, "W"), numpy_helper.from_array(b_fold, "B")]
c_f = helper.make_node("Conv", ["X","W","B"], ["Y"], pads=[1,1,1,1])
g_f = helper.make_graph([c_f], "g", [X_f], [Y_f], initializer=inits_f)
m_f = helper.make_model(g_f, opset_imports=[helper.make_opsetid("", 11)])
folded_out = ort.InferenceSession(m_f.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

max_diff = float(np.max(np.abs(ref_out - folded_out)))
print(f"Conv+BN output[0,0,0,:3]:      {ref_out[0,0,0,:3]}")
print(f"Folded conv output[0,0,0,:3]:  {folded_out[0,0,0,:3]}")
print(f"Max abs diff (correct fold ~0): {max_diff:.8f}")
print(f"TF bug: quantization folding applied scale incorrectly, causing larger divergence")
PASS = max_diff < 1e-3
print(f"PASS={PASS}")
