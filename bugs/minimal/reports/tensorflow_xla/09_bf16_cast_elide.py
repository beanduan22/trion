"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — TensorFlow XLA: bf16↔fp32 cast round-trip eliminated as id     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : TensorFlow 2.21 + XLA (jit_compile=True)
Related GitHub    : openxla/xla — same class of bug as pytorch/pytorch#179561
Severity          : High (silent up-precision of mixed-precision models)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
`tf.cast(tf.cast(x, tf.bfloat16), tf.float32)` is a LOSSY precision
truncation in eager mode.  Under `tf.function(jit_compile=True)` (XLA),
both Cast nodes are stripped by XLA's algebraic simplifier, producing
the original fp32 tensor unchanged.

This is the same bug as PyTorch Inductor #179561 but in the XLA backend.
Because JAX also uses XLA, this bug is shared between TF-XLA and JAX-jit —
a single XLA fix resolves both.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
XLA's AlgebraicSimplifier pass has a rule that eliminates adjacent Convert
(cast) ops when the source and destination types match:
    Convert(T → U) → Convert(U → T)    reduces to    no-op.
The rule is applied without verifying that U has ≥ mantissa and ≥ exponent
bits of T.  For bf16↔fp32 this is wrong because bf16 has 7 mantissa bits vs
fp32's 23.

Location (likely): `xla/service/algebraic_simplifier.cc`, the rule that
fires on back-to-back `HloInstruction::kConvert` nodes.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
  TF eager       → 1.234375 ✓ truncated
  TF-XLA         → 1.234568 ✗ ELIMINATED (bug)
  JAX eager      → 1.234375 ✓
  JAX-jit (XLA)  → 1.234568 ✗ ELIMINATED (same XLA cause)
  PyTorch eager  → 1.234375 ✓
  Inductor       → 1.234568 ✗ ELIMINATED (pytorch/pytorch#179561)
  ORT / OV / numpy / ONNX-Ref → 1.234375 ✓

Exit 0 = bug reproduces   Exit 1 = does not reproduce   Exit 2 = deps missing
"""
import sys
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
except ImportError:
    print("SKIP: tensorflow not installed"); sys.exit(2)

print(f"TensorFlow version: {tf.__version__}")

x_val = 1.234567890123
x = tf.constant([x_val], dtype=tf.float32)

def cast_roundtrip(t):
    return tf.cast(tf.cast(t, tf.bfloat16), tf.float32)

eager_out = float(cast_roundtrip(x).numpy()[0])

xla_fn = tf.function(cast_roundtrip, jit_compile=True)
xla_out = float(xla_fn(x).numpy()[0])

print(f"Input          : {x_val:.12f}")
print(f"Eager          : {eager_out:.12f}   (bf16 truncation applied — correct)")
print(f"XLA (jit=True) : {xla_out:.12f}   (Casts eliminated)")
print(f"eager diff     : {abs(x_val - eager_out):.2e}   ← bf16 rounding error")
print(f"XLA diff       : {abs(x_val - xla_out):.2e}   ← should match eager")

cast_elided = abs(xla_out - x_val) < 1e-5
eager_different = abs(eager_out - x_val) > 1e-5

if cast_elided and eager_different:
    print("\n✗ BUG — TF-XLA eliminated bf16↔fp32 cast pair as identity.")
    print("        Fix: in xla/service/algebraic_simplifier.cc, reject")
    print("        the 'Convert∘Convert → no-op' rule when U is narrower than T.")
    sys.exit(0)
print("\n✓ OK"); sys.exit(1)
