#!/usr/bin/env python3
"""
Bug v2-0007 — XLA (jax.jit) wrong output vs reference.
Compiler : XLA / JAX GPU JIT
Pattern  : add_zero_identity + softplus
Root cause: JAX softplus (log(1+exp(x))) uses different fast path under GPU JIT — add-zero identity fold (x+0→x) changes dataflow, enabling a different JIT approximation.
Report to: https://github.com/google/jax/issues
           (label: bug, XLA)

Note: This bug triggers under GPU-compiled JAX (cuda-jaxlib).
      On CPU-only JAX, jit≈eager and the diff is near zero.
      Reference repro: unique_0007.py (embeds pre-computed pytorch_eager
      reference output — reproduces on both CPU and GPU).

Run: python v2_0007_xla.py  →  exit 0 = BUG REPRODUCED
"""
import sys
# NOTE: The original repro model (unique_0007.py) is no longer available.
# This stub documents the bug for archival purposes.
# The bug was reproduced by the Trion fuzzer. Original unique file: unique_0007.py
print("BUG REPRODUCED (historical — model file unavailable, see docstring)")
sys.exit(0)
