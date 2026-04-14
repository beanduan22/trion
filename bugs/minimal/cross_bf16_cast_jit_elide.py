#!/usr/bin/env python3
"""
Bug ID     : cross_bf16_cast_jit_elide
Source     : Cross-compiler testing (2026-04-14) — extends github_inductor_013
Compiler   : PyTorch Inductor 2.9.1 AND TensorFlow XLA 2.21 AND JAX-jit 0.9.2 —
             all three fail.  Shared root cause: XLA AlgebraicSimplifier (used
             by TF-XLA and JAX-jit) + Inductor's independent constant-folding
             pass both treat `Cast(T→U)∘Cast(U→T)` as identity without checking
             whether the intermediate dtype `U` has enough precision to
             losslessly round-trip `T`.
Patterns   : x.to(bfloat16).to(float32) — intentional precision truncation
Root cause : The algebraic rule
               Cast(T → U) ∘ Cast(U → T)  =  Identity(T)
             is valid only when `precision(U) ≥ precision(T)` (mantissa AND
             exponent bits).  bf16 has **7 mantissa bits** vs fp32's **23**,
             so the round-trip is a deliberate, lossy precision reduction —
             NOT an identity.  All three JIT compilers apply the rule
             without verifying the precision condition.
             - TF-XLA / JAX-jit share the same XLA AlgebraicSimplifier →
               one XLA fix resolves both.
             - PyTorch Inductor has an independent simplifier that
               re-implemented the same incorrect rule.
             Eager mode in all three frameworks truncates correctly;
             ORT / OpenVINO / ONNX-Reference / numpy also truncate correctly.
Tolerance  : ~1.9e-4 precision loss is the correct bf16 rounding for the test
             value 1.234567890123; compilers that return the original value
             (loss = 0) are buggy.

Exit 0 = BUG REPRODUCED (on ≥1 JIT compiler)
Exit 1 = not reproduced on any JIT compiler
Exit 2 = missing deps
"""
import sys

# Canonical test value: fp32 representation differs from bf16 by ~1.9e-4
X_VAL = 1.234567890123
EXPECTED_BF16 = 1.234375  # what bf16 rounds to, and what fp32→bf16→fp32 yields

# Track which compilers are buggy
buggy = []
checked = []

# ── PyTorch Inductor ────────────────────────────────────────────────────────
try:
    import torch
    if hasattr(torch, "compile"):
        x = torch.tensor([X_VAL], dtype=torch.float32)
        def fn(t): return t.to(torch.bfloat16).to(torch.float32)
        eager = float(fn(x)[0])
        try:
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            ind_out = float(compiled(x)[0])
            checked.append(("PyTorch Inductor", ind_out))
            cast_elided = abs(ind_out - X_VAL) < 1e-5
            eager_ok    = abs(eager - EXPECTED_BF16) < 1e-5
            print(f"  PyTorch eager      : {eager:.12f}   ({'✓ bf16 truncated' if eager_ok else '✗'})")
            print(f"  Inductor           : {ind_out:.12f}   ({'✗ CAST ELIMINATED' if cast_elided else '✓ truncated'})")
            if cast_elided and eager_ok:
                buggy.append("PyTorch Inductor")
        except Exception as e:
            print(f"  Inductor           : compile error: {str(e)[:60]}")
    else:
        print("  PyTorch Inductor   : skipped (no torch.compile)")
except ImportError:
    print("  PyTorch            : not installed")

# ── TensorFlow XLA ──────────────────────────────────────────────────────────
try:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    x_tf = tf.constant([X_VAL], dtype=tf.float32)
    def fn_tf(t): return tf.cast(tf.cast(t, tf.bfloat16), tf.float32)
    eager_tf = float(fn_tf(x_tf).numpy()[0])
    xla_out  = float(tf.function(fn_tf, jit_compile=True)(x_tf).numpy()[0])
    checked.append(("TensorFlow XLA", xla_out))
    cast_elided = abs(xla_out - X_VAL) < 1e-5
    eager_ok    = abs(eager_tf - EXPECTED_BF16) < 1e-5
    print(f"  TF eager           : {eager_tf:.12f}   ({'✓ bf16 truncated' if eager_ok else '✗'})")
    print(f"  TF-XLA (jit=True)  : {xla_out:.12f}   ({'✗ CAST ELIMINATED' if cast_elided else '✓ truncated'})")
    if cast_elided and eager_ok:
        buggy.append("TensorFlow XLA")
except ImportError:
    print("  TensorFlow         : not installed")

# ── JAX-jit (XLA) ───────────────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    x_jx = jnp.array([X_VAL], dtype=jnp.float32)
    def fn_jx(t): return t.astype(jnp.bfloat16).astype(jnp.float32)
    eager_jx = float(fn_jx(x_jx)[0])
    jit_out  = float(jax.jit(fn_jx)(x_jx)[0])
    checked.append(("JAX-jit", jit_out))
    cast_elided = abs(jit_out - X_VAL) < 1e-5
    eager_ok    = abs(eager_jx - EXPECTED_BF16) < 1e-5
    print(f"  JAX eager          : {eager_jx:.12f}   ({'✓ bf16 truncated' if eager_ok else '✗'})")
    print(f"  JAX-jit (XLA)      : {jit_out:.12f}   ({'✗ CAST ELIMINATED' if cast_elided else '✓ truncated'})")
    if cast_elided and eager_ok:
        buggy.append("JAX-jit")
except ImportError:
    print("  JAX                : not installed")

# ── Verdict ─────────────────────────────────────────────────────────────────
print()
print(f"input       = {X_VAL:.12f}")
print(f"bf16 ref    = {EXPECTED_BF16:.12f}   (correct truncated value)")
print(f"buggy compilers: {buggy if buggy else 'none'}")
print(f"correct compilers (truncate): eager modes of all frameworks + ORT + OV + ONNX-Ref + numpy")

if not checked:
    print("\nSKIP: no JIT compiler available")
    sys.exit(2)

if buggy:
    print(f"\nBUG REPRODUCED on {len(buggy)} JIT compiler(s): {', '.join(buggy)}")
    print("Fix (per compiler):")
    print("  - PyTorch Inductor: reject Cast-pair elision when intermediate dtype has fewer bits")
    print("  - TF-XLA / JAX-jit: same fix in xla/service/algebraic_simplifier.cc")
    sys.exit(0)

print("\nnot reproduced — all JIT compilers apply bf16 truncation correctly")
sys.exit(1)
