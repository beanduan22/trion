#!/usr/bin/env python3
"""
Bug ID     : cross_xla_autocluster_slice_oob_suppress
Source     : Cross-framework testing (2026-04-16)
Compiler   : TensorFlow XLA auto-clustering (TF_XLA_FLAGS=--tf_xla_auto_jit=2
             --tf_xla_cpu_global_jit) — autocluster only. TF eager,
             @tf.function (no XLA), and @tf.function(jit_compile=True) all
             raise the expected InvalidArgumentError.
Patterns   : @tf.function with two tf.slice calls where sliced2 is OOB but
             never returned; sliced1 is valid and returned.
Root cause : The autocluster pass dead-code-eliminates `sliced2` before
             running bounds-checking on the Slice op, so the invalid
             `size[2]=4` on a dimension of size 3 is never validated.
             Eager and jit_compile=True both run the Slice kernel (or its
             XLA lowering) and correctly raise InvalidArgumentError.
             The key asymmetry: jit_compile validates even unused ops;
             autocluster prunes the graph before lowering.
             Reproduces across all tested 4D shapes (dim 2 always = 3):
               (1,3,3,2) / (2,4,3,3) / (1,5,3,1) / (3,2,3,4)

Exit 0 = BUG REPRODUCED on autocluster for at least one case
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import os, sys

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

@tf.function
def f_autocluster(t):
    sliced1 = tf.slice(t, [0, 0, 1, 0], [-1, -1, 1, -1])  # valid
    sliced2 = tf.slice(t, [0, 1, 0, 0], [-1, -1, 4, -1])  # OOB: size=4 on dim of size 3
    return sliced1                                           # sliced2 never used

os.environ['TF_XLA_FLAGS'] = ''

@tf.function(jit_compile=True)
def f_jit(t):
    sliced1 = tf.slice(t, [0, 0, 1, 0], [-1, -1, 1, -1])
    sliced2 = tf.slice(t, [0, 1, 0, 0], [-1, -1, 4, -1])
    return sliced1

def f_eager(t):
    sliced1 = tf.slice(t, [0, 0, 1, 0], [-1, -1, 1, -1])
    sliced2 = tf.slice(t, [0, 1, 0, 0], [-1, -1, 4, -1])
    return sliced1

CASES = [
    ("V1: shape=(1,3,3,2)", (1,3,3,2)),
    ("V2: shape=(2,4,3,3)", (2,4,3,3)),
    ("V3: shape=(1,5,3,1)", (1,5,3,1)),
    ("V4: shape=(3,2,3,4)", (3,2,3,4)),
]

any_bug = False
print(f"{'Case':<22} {'Eager':<30} {'XLA jit':<30} {'Autocluster'}")
print("-" * 100)
for label, shape in CASES:
    n = int(np.prod(shape))
    x = tf.cast(tf.reshape(tf.range(n), shape), tf.float32)

    def run(fn):
        try:
            out = fn(x)
            return f"OK  {out.numpy().flatten()[:3].tolist()}..."
        except Exception as e:
            return f"RAISED {type(e).__name__}"

    r_eager = run(f_eager)
    r_jit   = run(f_jit)
    r_auto  = run(f_autocluster)
    bug = "RAISED" not in r_auto and "RAISED" in r_eager
    if bug:
        any_bug = True
    print(f"{label:<22} {r_eager:<30} {r_jit:<30} {r_auto}{'  ← BUG' if bug else ''}")

sys.exit(0 if any_bug else 1)
