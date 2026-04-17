#!/usr/bin/env python3
"""
Bug ID     : github_tf_61881
Source     : https://github.com/tensorflow/tensorflow/issues/61881
Compiler   : TensorFlow XLA (jit_compile=True)
Patterns   : tf.matmul with incompatible inner dimensions under XLA
Root cause : XLA silently accepts a matrix multiply where the inner dimensions
             don't match (e.g. [1,4] @ [6,1]) and returns a result, while
             TF eager correctly raises InvalidArgumentError.
             The XLA behavior is dangerous: it reads garbage memory and returns
             a plausible-looking tensor without any warning.
Test case  : x=[1,4], w=[6,1] — inner dims 4≠6 → invalid matmul
             XLA  => returns a tensor (BUG)
             Eager => raises InvalidArgumentError (correct)
Exit 0 = bug reproduced (XLA succeeds on invalid matmul; eager raises error)
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import sys

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

# weight shape [6,1]; input [1,4] → inner dims 4≠6 → invalid
W_INIT = tf.constant(
    [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], dtype=tf.float32
)
X = tf.constant([[2., 4., 6., 8.]], dtype=tf.float32, shape=[1, 4])

# ── Eager (baseline) ─────────────────────────────────────────────────────────
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

# ── XLA ──────────────────────────────────────────────────────────────────────
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

# ── Report ───────────────────────────────────────────────────────────────────
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
