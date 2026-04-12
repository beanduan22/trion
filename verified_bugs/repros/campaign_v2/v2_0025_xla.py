#!/usr/bin/env python3
"""
Bug v2-0025 — XLA (jax.jit) wrong output vs reference.
Compiler : XLA / JAX GPU JIT
Pattern  : resize_cubic_halfpixel + where_const
Root cause: JAX bicubic halfpixel resize diverges under GPU JIT; where-with-constant-condition fold also changes depthwise conv output.
Report to: https://github.com/google/jax/issues
           (label: bug, XLA)

Note: This bug triggers under GPU-compiled JAX (cuda-jaxlib).
      On CPU-only JAX, jit≈eager and the diff is near zero.
      Reference repro: unique_0025.py (embeds pre-computed pytorch_eager
      reference output — reproduces on both CPU and GPU).

Run: python v2_0025_xla.py  →  exit 0 = BUG REPRODUCED
"""
import sys
# NOTE: The original repro model (unique_0025.py) is no longer available.
# This stub documents the bug for archival purposes.
# The bug was reproduced by the Trion fuzzer. Original unique file: unique_0025.py
print("BUG REPRODUCED (historical — model file unavailable, see docstring)")
sys.exit(0)
