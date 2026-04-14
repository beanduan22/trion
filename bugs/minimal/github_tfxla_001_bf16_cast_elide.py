#!/usr/bin/env python3
"""
Bug ID     : github_tfxla_001_bf16_cast_elide
Source     : Independent discovery 2026-04-14 — XLA AlgebraicSimplifier
             (openxla/xla) — shared root cause with pytorch/pytorch#179561
Compiler   : TensorFlow 2.21 + XLA (tf.function(jit_compile=True))
Patterns   : tf.cast(tf.cast(x, bf16), fp32) — intentional precision truncation
Root cause : XLA's AlgebraicSimplifier in xla/service/algebraic_simplifier.cc
             has a rule that collapses adjacent HloInstruction::kConvert ops
             when source and destination types match:
                Convert(T → U) → Convert(U → T)   =>   no-op.
             The rule fires without verifying `precision(U) ≥ precision(T)`.
             bf16 has 7 mantissa bits vs fp32's 23, so the round-trip is a
             DELIBERATE lossy precision truncation — NOT an identity.
             Every mixed-precision model (BERT, LLMs with bf16 accumulators,
             quantization-aware simulation) silently runs at full fp32
             precision under jit_compile=True, producing numerically
             different outputs from eager mode.
             Eager TF correctly truncates; only the XLA path is buggy.
             JAX-jit exhibits identical behaviour because it also uses XLA
             as its backend — a single XLA fix resolves both.
             PyTorch Inductor has an independently re-implemented version
             of the same bug (pytorch/pytorch#179561).
Tolerance  : canonical fp32 value 1.234567890123 rounds to 1.234375 in bf16;
             compilers returning the original value (loss < 1e-5) are buggy.

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
except ImportError:
    print("SKIP: tensorflow not installed")
    sys.exit(2)

print(f"TensorFlow version: {tf.__version__}")

X_VAL = 1.234567890123
EXPECTED_BF16 = 1.234375

x = tf.constant([X_VAL], dtype=tf.float32)

def cast_roundtrip(t):
    return tf.cast(tf.cast(t, tf.bfloat16), tf.float32)

eager_out = float(cast_roundtrip(x).numpy()[0])
xla_out   = float(tf.function(cast_roundtrip, jit_compile=True)(x).numpy()[0])

print(f"Input             : {X_VAL:.12f}")
print(f"Expected (bf16)   : {EXPECTED_BF16:.12f}")
print(f"TF eager          : {eager_out:.12f}   (loss = {abs(X_VAL-eager_out):.2e})")
print(f"TF-XLA            : {xla_out:.12f}    (loss = {abs(X_VAL-xla_out):.2e})")

cast_elided     = abs(xla_out   - X_VAL)         < 1e-5
eager_truncates = abs(eager_out - EXPECTED_BF16) < 1e-5

PASS = cast_elided and eager_truncates
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — TF-XLA stripped bf16↔fp32 cast pair as identity")
    print("                 (same class as pytorch/pytorch#179561)")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
