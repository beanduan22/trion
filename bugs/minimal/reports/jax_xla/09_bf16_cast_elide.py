"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — JAX / XLA: bf16↔fp32 cast round-trip eliminated as identity    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : JAX 0.9.2 + XLA backend (jax.jit)
Related project   : openxla/xla (shared backend with TF-XLA)
                    Also seen in pytorch/pytorch#179561 (Inductor)
Severity          : High (silent up-precision of mixed-precision models)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
`x.astype(jnp.bfloat16).astype(jnp.float32)` is a LOSSY precision truncation
when evaluated eagerly.  Under `jax.jit` (XLA), the AlgebraicSimplifier pass
strips both Cast ops, returning the original fp32 tensor unchanged.

Shared root cause with TF-XLA (same XLA backend): a single XLA fix resolves
both.  PyTorch Inductor has the same bug in its own simplifier.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
XLA's `AlgebraicSimplifier::HandleConvert` (or equivalent rewrite rule in
`algebraic_simplifier.cc`) collapses a pair of `Convert` HLO instructions
whose endpoint types match, without checking whether the intermediate type
can losslessly represent the source.  The condition must be:

    precision(U) ≥ precision(T)   (mantissa AND exponent bits)

For bf16 vs fp32: mantissa 7 vs 23 → rule should NOT fire.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: x = 1.234567890123  (fp32)
bf16 representation: 1.234375

  JAX eager      → 1.234375 ✓ truncated (correct)
  JAX-jit (XLA)  → 1.234568 ✗ ELIMINATED
  TF eager       → 1.234375 ✓
  TF-XLA         → 1.234568 ✗ ELIMINATED (same XLA cause)
  PyTorch eager  → 1.234375 ✓
  Inductor       → 1.234568 ✗ ELIMINATED (pytorch/pytorch#179561)
  ORT / OV / numpy / ONNX-Ref → 1.234375 ✓

Exit 0 = bug reproduces   Exit 1 = does not reproduce   Exit 2 = deps missing
"""
import sys

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("SKIP: jax not installed"); sys.exit(2)

print(f"JAX version: {jax.__version__}")

x_val = 1.234567890123
x = jnp.array([x_val], dtype=jnp.float32)

def cast_roundtrip(t):
    return t.astype(jnp.bfloat16).astype(jnp.float32)

eager_out = float(cast_roundtrip(x)[0])
jit_out   = float(jax.jit(cast_roundtrip)(x)[0])

print(f"Input      : {x_val:.12f}")
print(f"Eager      : {eager_out:.12f}   (bf16 rounding applied — correct)")
print(f"JIT (XLA)  : {jit_out:.12f}   (Casts eliminated)")
print(f"eager loss : {abs(x_val - eager_out):.2e}")
print(f"jit loss   : {abs(x_val - jit_out):.2e}   (should match eager loss)")

cast_elided = abs(jit_out - x_val) < 1e-5
eager_different = abs(eager_out - x_val) > 1e-5

if cast_elided and eager_different:
    print("\n✗ BUG — jax.jit (XLA) eliminated bf16↔fp32 cast pair as identity.")
    print("        Fix is in openxla/xla AlgebraicSimplifier — must reject the")
    print("        'Convert∘Convert → no-op' rule when U is narrower than T.")
    sys.exit(0)
print("\n✓ OK"); sys.exit(1)
