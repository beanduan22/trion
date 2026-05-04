import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

W_INIT = tf.constant(
    [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], dtype=tf.float32
)
X = tf.constant([[2., 4., 6., 8.]], dtype=tf.float32, shape=[1, 4])

class ModelEager(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(W_INIT, shape=tf.TensorShape(None), dtype='float32')
    @tf.function
    def call(self, x):
        return tf.matmul(x, self.w)

eager_out = None
eager_err = None
try:
    m_eager = ModelEager()
    eager_out = m_eager(X).numpy()
except Exception as e:
    eager_err = type(e).__name__ + ": " + str(e)[:80]

class ModelXLA(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(W_INIT, shape=tf.TensorShape(None), dtype='float32')
    @tf.function(jit_compile=True)
    def call(self, x):
        return tf.matmul(x, self.w)

xla_out = None
xla_err = None
try:
    m_xla = ModelXLA()
    xla_out = m_xla(X).numpy()
except Exception as e:
    xla_err = type(e).__name__ + ": " + str(e)[:80]

print(f"Input shape : {tuple(X.shape)}, W shape: {tuple(W_INIT.shape)}")
print(f"Eager output: {eager_out}  error={eager_err}")
print(f"XLA   output: {xla_out}  error={xla_err}")

xla_succeeded = xla_out is not None
eager_failed  = eager_err is not None

if eager_failed and xla_succeeded:
    print("BUG REPRODUCED: XLA silently executes invalid matmul; eager raises error.")
    sys.exit(0)
else:
    print("NOT REPRODUCED on current TF version.")
    sys.exit(1)
