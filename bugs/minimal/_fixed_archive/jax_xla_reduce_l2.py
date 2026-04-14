#!/usr/bin/env python3
"""
Bug ID     : jax_xla_reduce_l2
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : reduce_l2_last
Root cause : JAX XLA JIT applies a fast RSqrt approximation for L2 normalization that differs numerically from eager
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

x = np.random.randn(2, 4, 16).astype(np.float32)
x_jax = jnp.array(x)

# Eager reference: sqrt(sum(x^2, axis=-1))
ref = jax.block_until_ready(jnp.sqrt(jnp.sum(x_jax ** 2, axis=-1)))

# JIT compiled — XLA may substitute fast RSqrt approximation
l2_jit = jax.jit(lambda a: jnp.sqrt(jnp.sum(a ** 2, axis=-1)))
out = jax.block_until_ready(l2_jit(x_jax))

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
