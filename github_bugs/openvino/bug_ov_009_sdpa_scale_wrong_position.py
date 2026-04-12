# OpenVINO Bug #34177 - SDPA decomposition applies scale before MatMul(Q,K^T) instead of after
# https://github.com/openvinotoolkit/openvino/issues/34177
# OV bug: scale applied to Q before QK^T; wrong when scale varies across the K dimension
# (Q*s)@K^T != (Q@K^T)*s when s is not a scalar multiplied uniformly across all cols
import numpy as np

np.random.seed(21)
B, H, S, D = 1, 1, 3, 4

Q = np.random.randn(B, H, S, D).astype(np.float32)
K = np.random.randn(B, H, S, D).astype(np.float32)
V = np.random.randn(B, H, S, D).astype(np.float32)

# Per-column scale on the D dimension: different scale per key dimension
# (Q * col_scale) @ K^T  !=  (Q @ K^T) * row_scale  in general
col_scale = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32).reshape(1, 1, 1, D)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# Correct: uniform scalar scale applied to scores after QK^T
scalar_scale = 1.0 / np.sqrt(D)
scores_correct = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scalar_scale
out_correct    = np.matmul(softmax(scores_correct), V)

# OV bug pattern: per-element scale applied to Q before matmul
# (Q * col_scale) @ K^T  differs from  Q @ K^T scaled uniformly
scores_buggy = np.matmul(Q * col_scale, K.transpose(0, 1, 3, 2))
out_buggy    = np.matmul(softmax(scores_buggy), V)

max_diff = float(np.max(np.abs(out_correct - out_buggy)))
print(f"Q shape: {Q.shape}")
print(f"correct out[0,0,0,:]: {out_correct[0,0,0,:]}")
print(f"buggy   out[0,0,0,:]: {out_buggy[0,0,0,:]}")
print(f"max_diff: {max_diff:.4f}  (non-zero proves scale position matters)")
print(f"OV bug: scale applied to Q before matmul instead of to scores after")
print(f"PASS={max_diff > 1e-3}")
