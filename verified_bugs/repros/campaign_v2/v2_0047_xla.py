#!/usr/bin/env python3
"""
Bug v2-0047 — XLA (jax.jit) wrong output vs reference.
Compiler  : XLA / JAX GPU JIT
Source    : bug_000651 (model #651)
Patterns  : l2_norm_manual_primitives, div_by_constant, mul_add_mul_chain, min_variadic_4inputs, manual_layernorm
Root cause: XLA GPU JIT incorrectly fuses manual L2-norm with div-by-constant and min-over-4-inputs — XLA algebraic simplification cancels div-by-constant before L2-norm is computed correctly.
Report to : https://github.com/google/jax/issues
Note      : XLA bugs require GPU-compiled JAX (cuda-jaxlib).
            On CPU-only JAX, jit ≈ eager and diff is near zero.

Oracle: XLA uses pytorch_eager (onnx2torch) as reference via ONNX model.
ONNX model: /home/binduan/myspace/trion/trion_results/bug_000651.onnx
Run: python v2_0047_xla.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os, subprocess

# Delegate to the primary JAX repro file which has the XLA oracle
_ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'unique_2666.py')

if not os.path.exists(_ref):
    print(f"Reference unique_2666.py not found")
    sys.exit(1)

r = subprocess.run([sys.executable, _ref], capture_output=True, timeout=120)
sys.stdout.buffer.write(r.stdout)
sys.stderr.buffer.write(r.stderr)
if r.returncode == 0 and b'BUG REPRODUCED' in r.stdout:
    sys.exit(0)
sys.exit(r.returncode)
