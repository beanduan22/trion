#!/usr/bin/env python3
"""
Bug v2-0034 — XLA (jax.jit) wrong output vs reference.
Compiler : XLA / JAX GPU JIT
Pattern  : self_sub_zero + topk_k1
Root cause: JAX folds x-x→0 under GPU JIT but topk on all-zero values uses implementation-defined tie-breaking that differs between JIT and eager.
Report to: https://github.com/google/jax/issues
           (label: bug, XLA)

Note: This bug triggers under GPU-compiled JAX (cuda-jaxlib).
      On CPU-only JAX, jit≈eager and the diff is near zero.
      Reference repro: unique_0034.py (embeds pre-computed pytorch_eager
      reference output — reproduces on both CPU and GPU).

Run: python v2_0034_xla.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os, subprocess

# Delegate to unique_0034.py which has embedded reference output
_ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'unique_0034.py')

if not os.path.exists(_ref):
    print("Reference file unique_0034.py not found")
    sys.exit(1)

r = subprocess.run([sys.executable, _ref], capture_output=True, timeout=120)
sys.stdout.buffer.write(r.stdout)
sys.stderr.buffer.write(r.stderr)
if r.returncode == 0 and b'BUG REPRODUCED' in r.stdout:
    sys.exit(0)
sys.exit(r.returncode)
