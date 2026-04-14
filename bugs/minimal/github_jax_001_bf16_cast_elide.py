#!/usr/bin/env python3
"""
Bug ID     : github_jax_001_bf16_cast_elide
Source     : Independent discovery 2026-04-14 — openxla/xla AlgebraicSimplifier
             (shared XLA backend with TF-XLA; same class as
             pytorch/pytorch#179561 Inductor)
Compiler   : JAX 0.9.2 + XLA (jax.jit)
Patterns   : x.astype(bfloat16).astype(float32) — intentional precision
             truncation
Root cause : jax.jit lowers to XLA HLO; XLA's AlgebraicSimplifier eliminates
             the adjacent Convert-pair as an identity without verifying
             `precision(U) ≥ precision(T)`.  bf16 has 7 mantissa bits, fp32
             has 23 — so the round-trip is lossy and must not be eliminated.
             Eager JAX correctly applies the truncation; only jax.jit is
             buggy.  TF-XLA exhibits identical behaviour via the same
             simplifier pass; one XLA fix resolves both.
             The same algebraic error has been independently re-implemented
             in PyTorch Inductor (pytorch/pytorch#179561).
             Impact: any mixed-precision JAX model (Flax BERT/T5/LLMs with
             bf16 accumulators) silently runs at full fp32 precision under
             jax.jit, producing numerically different outputs vs eager.
Tolerance  : canonical fp32 value 1.234567890123 → bf16 = 1.234375.  Any
             jit-compiled output with loss < 1e-5 from the original fp32
             value indicates the cast pair was eliminated.

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("SKIP: jax not installed")
    sys.exit(2)

print(f"JAX version: {jax.__version__}")

X_VAL = 1.234567890123
EXPECTED_BF16 = 1.234375

x = jnp.array([X_VAL], dtype=jnp.float32)

def cast_roundtrip(t):
    return t.astype(jnp.bfloat16).astype(jnp.float32)

eager_out = float(cast_roundtrip(x)[0])
jit_out   = float(jax.jit(cast_roundtrip)(x)[0])

print(f"Input             : {X_VAL:.12f}")
print(f"Expected (bf16)   : {EXPECTED_BF16:.12f}")
print(f"JAX eager         : {eager_out:.12f}   (loss = {abs(X_VAL-eager_out):.2e})")
print(f"JAX-jit (XLA)     : {jit_out:.12f}    (loss = {abs(X_VAL-jit_out):.2e})")

cast_elided     = abs(jit_out   - X_VAL)         < 1e-5
eager_truncates = abs(eager_out - EXPECTED_BF16) < 1e-5

PASS = cast_elided and eager_truncates
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — jax.jit eliminated bf16↔fp32 cast pair as identity")
    print("                 (shared XLA cause with TF-XLA; same class as Inductor)")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
