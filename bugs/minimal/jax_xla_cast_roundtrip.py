#!/usr/bin/env python3
"""
Bug ID     : jax_xla_cast_roundtrip
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : cast_roundtrip
Root cause : JAX XLA JIT fuses the int32 cast pair into identity, losing the floor-truncation; values near 0.5 diverge
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

# Values near 0.5 boundaries to expose truncation differences
x_np = np.array([0.1, 0.6, 1.4, -0.7, 2.9, -1.5, 0.5, -0.5], dtype=np.float32)
x_jax = jnp.array(x_np)

# Eager reference: float32 -> int32 -> float32
ref = jax.block_until_ready(x_jax.astype(jnp.int32).astype(jnp.float32))

# JIT compiled — XLA may fuse cast pair into identity
cast_jit = jax.jit(lambda a: a.astype(jnp.int32).astype(jnp.float32))
out = jax.block_until_ready(cast_jit(x_jax))

diff = float(jnp.max(jnp.abs(out - ref)))
TOL = 0.01
PASS = diff <= TOL

print(f"max_diff={diff:.6f}  tol={TOL}  pass={PASS}")
print(f"ref={np.array(ref)}  out={np.array(out)}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
