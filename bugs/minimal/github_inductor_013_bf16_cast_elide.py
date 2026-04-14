#!/usr/bin/env python3
"""
Bug ID     : github_inductor_013_bf16_cast_elide
Source     : GitHub — pytorch/pytorch#179561 + independent verification 2026-04-14
Compiler   : PyTorch Inductor (torch.compile, backend=inductor)
Related    : Cross-compiler study 2026-04-14 found the SAME algebraic error in
             two other JIT compilers (TF-XLA, JAX-jit) — see sibling scripts
             github_tfxla_001_bf16_cast_elide.py and
             github_jax_001_bf16_cast_elide.py; see cross_bf16_cast_jit_elide.py
             for unified multi-compiler test.  All three apply
                 Cast(T→U) ∘ Cast(U→T)  =  Identity(T)
             without verifying precision(U) ≥ precision(T).  TF-XLA and JAX-jit
             share the openxla/xla AlgebraicSimplifier (one XLA fix resolves
             both).  PyTorch Inductor has an independent simplifier that
             re-implemented the same incorrect rule.
Patterns   : bfloat16→float32 cast round-trip precision
Root cause : Inductor's constant-folding / cast-elimination pass treats the chain
             x.to(bfloat16).to(float32) as a no-op identity and strips both Cast
             nodes from the IR.  In eager mode the cast to bfloat16 intentionally
             truncates the FP32 mantissa (BF16 has 7 bits vs FP32's 23), so the
             round-trip is a lossy precision reduction, NOT an identity.
             Any model that uses bf16 casting for intentional quantisation/precision
             control (e.g., mixed-precision transformers, bf16 accumulation layers)
             will silently compute at full fp32 precision instead, producing
             numerically different outputs than the eager reference.

             Observed:
               x         = 1.2345678806 (fp32)
               eager out = 1.2343750000 (bf16 rounding applied: 1.234375 = closest bf16)
               compiled  = 1.2345678806 (both casts eliminated — wrong)

Tolerance  : 1e-4 (bf16 precision loss is ~1.9e-4 for this value)

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys

try:
    import torch
except ImportError:
    print("SKIP: torch not installed")
    sys.exit(2)

if not hasattr(torch, 'compile'):
    print("SKIP: torch.compile not available (need PyTorch >= 2.0)")
    sys.exit(2)

# Precise FP32 value whose BF16 representation differs measurably
x = torch.tensor([1.234567890123456789], dtype=torch.float32)

def model_fn(t):
    """Intentional lossy precision cast: fp32 → bf16 → fp32"""
    return t.to(torch.bfloat16).to(torch.float32)

# Eager reference: bf16 truncates mantissa
ref = model_fn(x)

# Compiled: inductor incorrectly eliminates both casts
try:
    compiled = torch.compile(model_fn, backend="inductor", fullgraph=True)
    compiled_out = compiled(x)
except Exception as e:
    print(f"compile error: {e}")
    sys.exit(2)

val_original = float(x[0])
val_ref      = float(ref[0])
val_compiled = float(compiled_out[0])

print(f"Input x             : {val_original:.12f}")
print(f"Eager output        : {val_ref:.12f}  (bf16 rounding applied)")
print(f"Compiled output     : {val_compiled:.12f}")
print()
precision_loss   = abs(val_original - val_ref)
compiled_vs_ref  = abs(val_compiled - val_ref)
compiled_is_x    = abs(val_compiled - val_original) < 1e-10

print(f"Eager precision loss (fp32→bf16→fp32): {precision_loss:.2e}")
print(f"Compiled vs eager diff               : {compiled_vs_ref:.2e}")
print(f"Compiled output == original x        : {compiled_is_x}")

PASS = compiled_is_x and (precision_loss > 1e-5)
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — Inductor eliminates bf16↔fp32 cast pair as identity")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
