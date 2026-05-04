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

X = tf.constant(
    np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2),
    dtype=tf.float32,
)

class ModelEager(tf.keras.Model):
    @tf.function
    def call(self, x1):
        sliced1 = tf.slice(x1, [0, 0, 1, 0], [-1, -1, 1, -1])   # used
        sliced2 = tf.slice(x1, [0, 0, 0, 0], [-1, -1, 5, -1])   # dead — invalid!
        return sliced1

eager_out = None
eager_err = None
try:
    m_eager = ModelEager()
    eager_out = m_eager(X).numpy()
except Exception as e:
    eager_err = type(e).__name__ + ": " + str(e)[:80]

class ModelXLA(tf.keras.Model):
    @tf.function(jit_compile=True)
    def call(self, x1):
        sliced1 = tf.slice(x1, [0, 0, 1, 0], [-1, -1, 1, -1])   # used
        sliced2 = tf.slice(x1, [0, 0, 0, 0], [-1, -1, 5, -1])   # dead — invalid!
        return sliced1

xla_out = None
xla_err = None
try:
    m_xla = ModelXLA()
    xla_out = m_xla(X).numpy()
except Exception as e:
    xla_err = type(e).__name__ + ": " + str(e)[:80]

print(f"Input shape : {tuple(X.shape)}")
print(f"Eager output shape: {eager_out.shape if eager_out is not None else None}  error={eager_err}")
print(f"XLA   output shape: {xla_out.shape if xla_out is not None else None}  error={xla_err}")

eager_passed = eager_out is not None
xla_failed   = xla_err is not None

if eager_passed and xla_failed:
    print("BUG REPRODUCED: XLA executes dead slice (no DCE); eager succeeds.")
    sys.exit(0)
else:
    print("NOT REPRODUCED on current TF version.")
    sys.exit(1)
