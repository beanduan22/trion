#!/usr/bin/env python3
"""
Bug ID     : jax_xla_row_reduce_transpose
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : row_reduce_mul_transpose
Root cause : JAX XLA JIT reorders the transpose with the reduce+multiply fusion, producing wrong layout
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps

# Requires JAX GPU (cuda-jaxlib) to reproduce
"""

import sys as _sys

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
except ImportError:
    print("missing deps: jax not installed")
    _sys.exit(2)

# Check for GPU backend — this is a GPU-only bug
try:
    gpu_devices = jax.devices("gpu")
    if len(gpu_devices) == 0:
        raise RuntimeError("no GPU devices found")
except Exception:
    print("No JAX GPU backend found; skipping (CPU-only environment)")
    PASS = True
    print("not reproduced")
    _sys.exit(1)

np.random.seed(42)

# Use a 2D matrix so that .T is well-defined and sum(axis=1) is "row reduce"
x = np.random.randn(8, 6).astype(np.float32)
x_jax = jnp.array(x)

def row_reduce_mul_transpose(a):
    # ReduceSum across rows (axis=1), keepdims, multiply elementwise, then transpose
    row_sum = jnp.sum(a, axis=1, keepdims=True)   # [8, 1]
    scaled = row_sum * a                             # [8, 6]
    return scaled.T                                  # [6, 8]

# Eager reference
ref = jax.block_until_ready(row_reduce_mul_transpose(x_jax))

# JIT compiled
fn_jit = jax.jit(row_reduce_mul_transpose)
out = jax.block_until_ready(fn_jit(x_jax))

diff = float(jnp.max(jnp.abs(out - ref)))
TOL = 0.01
PASS = diff <= TOL

print(f"max_diff={diff:.6f}  tol={TOL}  pass={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
