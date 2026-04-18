#!/usr/bin/env python3
"""
Bug ID     : github_tf_61884
Source     : https://github.com/tensorflow/tensorflow/issues/61884
Compiler   : TensorFlow XLA (jit_compile=True) vs TF autocluster
Patterns   : Dead slice operation not eliminated by XLA but removed by autocluster
Root cause : XLA does not perform dead-code elimination (DCE) on tf.slice ops
             that are never used in the output. The dead slice specifies an
             invalid range that would fail at runtime, so XLA raises
             InvalidArgumentError. TF autocluster (auto_jit=2) correctly
             eliminates the dead code and the model runs fine.
Test case  : input shape [1, 2, 3, 2]
             sliced1 = tf.slice(x, [0, 0, 1, 0], [-1, -1, 1, -1])  <- returned
             sliced2 = tf.slice(x, [0, 0, 0, 0], [-1, -1, 5, -1])  <- dead (5>3 invalid)
             XLA executes sliced2 → InvalidArgumentError
             Eager skips dead code → no error
Exit 0 = bug reproduced (XLA raises error due to dead slice; eager succeeds)
Exit 1 = not reproduced
Exit 2 = missing deps
"""
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

# Input shape [1, 2, 3, 2]  — dim2=3; dead slice requests size 5 > 3
X = tf.constant(
    np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2),
    dtype=tf.float32,
)

# ── Eager (baseline) ─────────────────────────────────────────────────────────
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

# ── XLA ──────────────────────────────────────────────────────────────────────
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

# ── Report ───────────────────────────────────────────────────────────────────
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
