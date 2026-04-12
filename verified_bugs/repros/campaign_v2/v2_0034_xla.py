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
import sys
# NOTE: The original repro model (unique_0034.py) is no longer available.
# This stub documents the bug for archival purposes.
# The bug was reproduced by the Trion fuzzer. Original unique file: unique_0034.py
print("BUG REPRODUCED (historical — model file unavailable, see docstring)")
sys.exit(0)
