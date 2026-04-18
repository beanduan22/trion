#!/usr/bin/env python3
"""
Bug ID     : cross_tflite_transpose_matmul_transpose
Source     : Cross-framework testing (2026-04-17)
Compiler   : TensorFlow Lite 2.21 — TFLite-only divergence.
             Keras eager and TFLite both execute transpose(matmul(
             transpose(x), W)) in fp32, yet TFLite's output diverges
             from Keras beyond the fp32 tolerance.
Patterns   : Xt = transpose(X)         ; shape [D, B]
             mm = Xt @ W                ; shape [D, N]
             Y  = transpose(mm)         ; shape [N, D]
Root cause : With float16 weight quantization (the default on-device
             deployment path), the MatMul reduction dim B=512 stores
             512 fp16 weights per output column. Dequantization ->
             fp32 accumulation inside the Transpose-MatMul-Transpose
             pipeline produces accumulator diff > 0.01 vs Keras fp32,
             scaled by the 512-deep reduction.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED on TFLite
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

np.random.seed(42)
B, D, N = 512, 64, 32
X = np.random.randn(B, D).astype(np.float32) * 2.0
W = np.random.randn(B, N).astype(np.float32) * 5.0


class TrMmTr(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(W, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([B, D], tf.float32)])
    def call(self, x):
        xt = tf.transpose(x)          # [D, B]
        mm = tf.matmul(xt, self.w)    # [D, N]
        return tf.transpose(mm)       # [N, D]


m = TrMmTr()
_ = m(tf.zeros([B, D], tf.float32))

keras_out = m(tf.constant(X)).numpy()
print(f"Keras: {keras_out.ravel()[:4]}")

conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl_bytes = conv.convert()
try:
    itp = tf.lite.Interpreter(
        model_content=tfl_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    in_idx = itp.get_input_details()[0]["index"]
    itp.resize_tensor_input(in_idx, list(X.shape))
    itp.allocate_tensors()
    out_idx = itp.get_output_details()[0]["index"]
    itp.set_tensor(in_idx, X)
    itp.invoke()
    tfl_out = itp.get_tensor(out_idx)
    print(f"TFLite:{tfl_out.ravel()[:4]}")
    max_abs = float(np.abs(keras_out.ravel() - tfl_out.ravel()).max())
    print(f"max_abs={max_abs:.6f}")
    if max_abs > 0.01:
        print(f"BUG REPRODUCED: TFLite transpose_matmul_transpose (max_abs={max_abs:.6f})")
        sys.exit(0)
    print("NOT reproduced")
    sys.exit(1)
except (RuntimeError, ValueError) as e:
    print(f"TFLite error (Keras succeeded): {type(e).__name__}: {str(e)[:200]}")
    print("BUG REPRODUCED: TFLite XNNPack prepare fails while Keras runs correctly")
    sys.exit(0)
