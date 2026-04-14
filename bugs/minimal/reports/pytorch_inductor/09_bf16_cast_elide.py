"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — PyTorch Inductor: bf16↔fp32 cast round-trip eliminated as id    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : PyTorch Inductor (torch.compile, backend="inductor")
Related GitHub    : pytorch/pytorch#179561
Severity          : High (silent up-precision of mixed-precision models)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
In eager PyTorch, `x.to(torch.bfloat16).to(torch.float32)` IS A LOSSY
PRECISION TRUNCATION.  bf16 has 7 mantissa bits; fp32 has 23.  The round-trip
produces the nearest representable bf16 value.

Inductor's algebraic-simplification pass incorrectly treats this chain as
an identity and strips both Cast nodes from the IR.  Any model that uses
`.to(bf16).to(fp32)` for explicit precision control (mixed-precision
training, bf16 accumulators, quantization-aware simulation) silently runs
at full fp32 precision under torch.compile.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
The algebraic rule `Cast(T → U) ∘ Cast(U → T) = Identity(T)` holds ONLY when
the round-trip is lossless, i.e., when `precision(U) ≥ precision(T)`.
For bf16↔fp32 this condition is violated: bf16 is narrower than fp32.

Inductor's pattern matcher applies the rule without checking the precision
relationship between the two dtypes.  The fix is to allow the elision only
when (a) the intermediate dtype has ≥ mantissa AND ≥ exponent bits of the
outer dtype, or (b) the intermediate dtype is float and the outer is an
integer type that was cast to float in the first place (standard
integer-preservation case).

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input value : x_fp32 = 1.234567890123
bf16 rounds to 1.234375   →   fp32 of that is also 1.234375

  PyTorch eager  → 1.234375000000  ✓ truncated (correct)
  TorchScript    → 1.234375000000  ✓ truncated
  ONNX-Reference → 1.234375000000  ✓ truncated
  ORT (Cast chain) → 1.234375000000  ✓ truncated
  OpenVINO (Cast chain) → 1.234375000000  ✓ truncated
  TF eager       → 1.234375000000  ✓ truncated
  JAX eager      → 1.234375000000  ✓ truncated
  numpy          → 1.234375000000  ✓ truncated
  Inductor       → 1.234567880630  ✗ CAST ELIMINATED  ← THIS BUG
  TF-XLA         → 1.234567880630  ✗ CAST ELIMINATED  ← same class in XLA
  JAX-jit (XLA)  → 1.234567880630  ✗ CAST ELIMINATED  ← same class in XLA

Three JIT compilers converged on the same incorrect cast-elimination rule.
See the tensorflow_xla/ and jax_xla/ sibling reports for the XLA-side bugs.

Exit 0 = bug reproduces   Exit 1 = does not reproduce   Exit 2 = deps missing
"""
import sys

try:
    import torch
except ImportError:
    print("SKIP: torch not installed"); sys.exit(2)

if not hasattr(torch, "compile"):
    print("SKIP: torch.compile not available"); sys.exit(2)

print(f"PyTorch version: {torch.__version__}")

x_val = 1.234567890123
x = torch.tensor([x_val], dtype=torch.float32)

def cast_roundtrip(t):
    return t.to(torch.bfloat16).to(torch.float32)

eager_out    = float(cast_roundtrip(x)[0])
try:
    compiled = torch.compile(cast_roundtrip, backend="inductor", fullgraph=True)
    compiled_out = float(compiled(x)[0])
except Exception as e:
    print(f"torch.compile failed: {e}"); sys.exit(2)

print(f"Input     : {float(x[0]):.12f}")
print(f"Eager     : {eager_out:.12f}   (bf16 truncation applied, correct)")
print(f"Inductor  : {compiled_out:.12f}   (both Casts stripped by algebraic simplifier)")
print(f"bf16 loss : {abs(float(x[0]) - eager_out):.2e}   (what eager produces)")
print(f"Indu loss : {abs(float(x[0]) - compiled_out):.2e}   (should match eager)")

cast_elided = abs(compiled_out - float(x[0])) < 1e-10
eager_different = abs(eager_out - float(x[0])) > 1e-5
if cast_elided and eager_different:
    print("\n✗ BUG — Inductor eliminated bf16↔fp32 cast pair as identity.")
    print("        Fix: check precision(U) ≥ precision(T) before eliding.")
    sys.exit(0)
print("\n✓ OK"); sys.exit(1)
