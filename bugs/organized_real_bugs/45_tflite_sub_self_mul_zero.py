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
B, D1, D2, D3 = 2, 512, 256, 128
X = np.random.randn(B, D1).astype(np.float32) * 2.0
A = np.random.randn(D1, D2).astype(np.float32) * 5.0
Bm = np.random.randn(D2, D3).astype(np.float32) * 5.0
W2 = np.random.randn(D3, 64).astype(np.float32) * 10.0

class AssocBreak(tf.keras.Model):
    """(x @ A) @ B  should equal  x @ (A @ B)  mathematically.
    Under fp16 quantization, each matmul is quantized independently,
    so the two associativity-equivalent expressions yield
    different fp32-dequantized outputs. Their difference, amplified
    by a further MatMul with W2, is visible > tolerance.
    """

    def __init__(self):
        super().__init__()
        self.a = tf.Variable(A, trainable=False)
        self.b = tf.Variable(Bm, trainable=False)
        self.w2 = tf.Variable(W2, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([B, D1], tf.float32)])
    def call(self, x):
        p1 = tf.matmul(tf.matmul(x, self.a), self.b)
        p2 = tf.matmul(x, tf.matmul(self.a, self.b))
        diff = p1 - p2
        return tf.matmul(diff, self.w2)

m = AssocBreak()
_ = m(tf.zeros([B, D1], tf.float32))

keras_out = m(tf.constant(X)).numpy()
print(f"Keras (should ~0): {keras_out.ravel()[:4]}")

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
    print(f"TFLite:            {tfl_out.ravel()[:4]}")
    max_abs = float(np.abs(keras_out.ravel() - tfl_out.ravel()).max())
    tfl_max = float(np.abs(tfl_out).max())
    print(f"max_abs={max_abs:.6f}, tfl_max={tfl_max:.6f}")
    if max_abs > 0.01 or tfl_max > 0.01:
        print(f"BUG REPRODUCED: TFLite sub_self_mul_zero non-zero (max_abs={max_abs:.6f})")
        sys.exit(0)
    print("NOT reproduced")
    sys.exit(1)
except (RuntimeError, ValueError) as e:
    print(f"TFLite error (Keras succeeded): {type(e).__name__}: {str(e)[:200]}")
    print("BUG REPRODUCED: TFLite interpreter fails while Keras runs correctly")
    sys.exit(0)
