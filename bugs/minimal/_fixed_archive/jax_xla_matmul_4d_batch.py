#!/usr/bin/env python3
"""
Bug ID     : jax_xla_matmul_4d_batch
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : matmul_4d_batch
Root cause : JAX XLA JIT miscompiles batched 4D matmul; output diverges from jax.numpy.matmul eager mode
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

x = np.random.randn(2, 4, 8, 16).astype(np.float32)
w = np.random.randn(2, 4, 16, 8).astype(np.float32)

x_jax = jnp.array(x)
w_jax = jnp.array(w)

# Eager reference
ref = jax.block_until_ready(jnp.matmul(x_jax, w_jax))

# JIT compiled
matmul_jit = jax.jit(lambda a, b: jnp.matmul(a, b))
out = jax.block_until_ready(matmul_jit(x_jax, w_jax))

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
