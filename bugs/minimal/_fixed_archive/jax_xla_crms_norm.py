#!/usr/bin/env python3
"""
Bug ID     : jax_xla_crms_norm
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : crms_norm
Root cause : JAX XLA JIT fuses the RMS norm with subsequent operations, producing numerical divergence vs eager
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

# Shape [1, 4, 8, 8]: batch=1, channels=4, H=8, W=8
x = np.random.randn(1, 4, 8, 8).astype(np.float32)
x_jax = jnp.array(x)

EPS = 1e-6

def crms_norm(a):
    # Channelized RMS norm: normalize per spatial location across channel axis
    # x / sqrt(mean(x^2, axis=1, keepdims=True) + eps)
    rms = jnp.sqrt(jnp.mean(a ** 2, axis=1, keepdims=True) + EPS)
    return a / rms

# Eager reference
ref = jax.block_until_ready(crms_norm(x_jax))

# JIT compiled
crms_jit = jax.jit(crms_norm)
out = jax.block_until_ready(crms_jit(x_jax))

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
