#!/usr/bin/env python3
"""
Cross-compiler bug: BatchNorm fusion / inference-mode errors
=============================================================
Compilers affected : PyTorch Inductor (#100970, #100987), TVM (#6852), TensorFlow (#43882),
                     OpenVINO (#23539)
Shared root cause  : All four compilers have bugs around BatchNorm at inference time:
                     - Inductor #100970: torch.compile BatchNorm2d produced wrong output shape
                     - Inductor #100987: Conv2d+BN train mode wrong batch stats
                     - TVM #6852:  SimplifyInference skips BN node → uses batch stats at inference
                     - TF #43882:  Conv+BN folding applies scale incorrectly in QAT
                     - OV #23539:  Type mismatch in Multiply during BN decomposition (1D/2D input)
Status             : All bugs closed/fixed. This repro validates the correct BN eval-mode
                     behavior and Conv+BN folding math via ONNX/ORT.

Three sub-tests:
  A) BatchNorm eval mode: must use running mean/var, NOT batch statistics.
  B) Conv+BN folding: merged weights must give identical output to unfused model.
  C) BatchNorm 1D (2D input): decomposition must preserve correct dtypes.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(42)

# ── Test A: BN eval mode uses running mean/var (TVM #6852 pattern) ─────────────
x_A   = np.random.randn(2, 4, 4, 4).astype(np.float32)
scale = np.ones(4,         dtype=np.float32)
bias  = np.zeros(4,        dtype=np.float32)
r_mean = np.full(4, 0.5,  dtype=np.float32)   # non-trivial running mean
r_var  = np.full(4, 2.0,  dtype=np.float32)   # non-trivial running var

X_A = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 4, 4])
Y_A = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 4, 4])
inits_A = [numpy_helper.from_array(v, n) for v, n in
           zip([scale, bias, r_mean, r_var], ["scale","bias","mean","var"])]
node_A  = helper.make_node("BatchNormalization",
                           ["X","scale","bias","mean","var"], ["Y"],
                           epsilon=1e-5, momentum=0.9)
graph_A = helper.make_graph([node_A], "g", [X_A], [Y_A], initializer=inits_A)
model_A = helper.make_model(graph_A, opset_imports=[helper.make_opsetid("", 9)])
out_A   = ort.InferenceSession(model_A.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(None, {"X": x_A})[0]
# Reference: must use running mean/var
ref_A = (x_A - r_mean[None,:,None,None]) / np.sqrt(r_var[None,:,None,None] + 1e-5)
ref_A = ref_A * scale[None,:,None,None] + bias[None,:,None,None]
diff_A = float(np.max(np.abs(out_A - ref_A)))
ok_A   = diff_A < 1e-4
print(f"A) BN eval mode: max_diff={diff_A:.2e}  (TVM #6852 bug: batch stats used instead)")

# ── Test B: Conv+BN folding (TF #43882 pattern) ─────────────────────────────
w      = np.random.randn(8, 4, 3, 3).astype(np.float32)
b_conv = np.random.randn(8).astype(np.float32)
gamma  = np.random.rand(8).astype(np.float32) + 0.5
beta   = np.random.randn(8).astype(np.float32)
r_mean_B = np.random.randn(8).astype(np.float32)
r_var_B  = np.random.rand(8).astype(np.float32) + 0.1
x_B    = np.random.randn(1, 4, 6, 6).astype(np.float32)
eps    = 1e-5

# Unfused reference
X_B = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 6, 6])
Y_B = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
inits_B = [numpy_helper.from_array(v, n) for v, n in
           zip([w, b_conv, gamma, beta, r_mean_B, r_var_B],
               ["W","Bc","gamma","beta","rmean","rvar"])]
c_ref  = helper.make_node("Conv", ["X","W","Bc"], ["cv"], pads=[1,1,1,1])
bn_ref = helper.make_node("BatchNormalization",
                          ["cv","gamma","beta","rmean","rvar"], ["Y"], epsilon=eps)
g_B = helper.make_graph([c_ref, bn_ref], "g", [X_B], [Y_B], initializer=inits_B)
m_B = helper.make_model(g_B, opset_imports=[helper.make_opsetid("", 9)])
ref_B = ort.InferenceSession(m_B.SerializeToString(),
                             providers=["CPUExecutionProvider"]).run(None, {"X": x_B})[0]

# Correct folding math
std    = np.sqrt(r_var_B + eps)
scale_f = gamma / std
w_fold  = w * scale_f[:, None, None, None]
b_fold  = (b_conv - r_mean_B) * scale_f + beta

X_f = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 6, 6])
Y_f = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
g_f = helper.make_graph(
    [helper.make_node("Conv", ["X","W","B"], ["Y"], pads=[1,1,1,1])],
    "g", [X_f], [Y_f],
    initializer=[numpy_helper.from_array(w_fold, "W"), numpy_helper.from_array(b_fold, "B")],
)
m_f     = helper.make_model(g_f, opset_imports=[helper.make_opsetid("", 11)])
fold_B  = ort.InferenceSession(m_f.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(None, {"X": x_B})[0]
diff_B  = float(np.max(np.abs(ref_B - fold_B)))
ok_B    = diff_B < 1e-3
print(f"B) Conv+BN folding: max_diff={diff_B:.2e}  (TF #43882 bug: wrong scale in QAT folding)")

# ── Test C: BN 1D (2D input tensor) — OV #23539 type mismatch pattern ─────────
N, C  = 4, 8
x_C   = np.random.randn(N, C).astype(np.float32)
g1d   = np.ones(C,  dtype=np.float32)
b1d   = np.zeros(C, dtype=np.float32)
m1d   = np.zeros(C, dtype=np.float32)
v1d   = np.ones(C,  dtype=np.float32)

X_C = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C])
Y_C = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, C])
inits_C = [numpy_helper.from_array(v, n) for v, n in
           zip([g1d, b1d, m1d, v1d], ["sc","b","m","v"])]
node_C = helper.make_node("BatchNormalization",
                          ["X","sc","b","m","v"], ["Y"],
                          epsilon=1e-5, training_mode=0)
graph_C = helper.make_graph([node_C], "g", [X_C], [Y_C], initializer=inits_C)
model_C = helper.make_model(graph_C, opset_imports=[helper.make_opsetid("", 15)])
out_C   = ort.InferenceSession(model_C.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(None, {"X": x_C})[0]
ref_C   = (x_C - m1d) / np.sqrt(v1d + 1e-5) * g1d + b1d
diff_C  = float(np.max(np.abs(out_C - ref_C)))
ok_C    = diff_C < 1e-4
print(f"C) BN 1D (2D input): max_diff={diff_C:.2e}  (OV #23539: type mismatch in decomposition)")

PASS = ok_A and ok_B and ok_C
print(f"Bugs: Inductor #100970/#100987, TVM #6852, TF #43882, OV #23539")
print(f"PASS={PASS}")
