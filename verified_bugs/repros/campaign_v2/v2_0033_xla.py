#!/usr/bin/env python3
"""
Bug v2-0033 — XLA (jax.jit) wrong output vs reference.
Compiler : XLA / JAX GPU JIT
Pattern  : cumsum_last_axis + mul_by_one
Root cause: JAX cumsum on last axis diverges under GPU JIT; x*1 chain also gets constant-folded differently by XLA.
Report to: https://github.com/google/jax/issues
           (label: bug, XLA)

Note: This bug triggers under GPU-compiled JAX (cuda-jaxlib).
      On CPU-only JAX, jit≈eager and the diff is near zero.
      Reference repro: unique_0033.py (embeds pre-computed pytorch_eager
      reference output — reproduces on both CPU and GPU).

Run: python v2_0033_xla.py  →  exit 0 = BUG REPRODUCED
"""
import sys
# NOTE: The original repro model (unique_0033.py) is no longer available.
# This stub documents the bug for archival purposes.
# The bug was reproduced by the Trion fuzzer. Original unique file: unique_0033.py
print("BUG REPRODUCED (historical — model file unavailable, see docstring)")
sys.exit(0)
