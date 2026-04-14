#!/usr/bin/env python3
"""
Bug ID     : jax_xla_self_sub_zero
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : self_sub_zero
Root cause : XLA JIT folds X - X to a zero constant BEFORE evaluating X, but the downstream ops expect a proper tensor (not a constant scalar broadcast), causing shape or value divergence
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

x = np.random.randn(1, 4, 8, 8).astype(np.float32)
something = np.random.randn(1, 4, 8, 8).astype(np.float32)

x_jax = jnp.array(x)
something_jax = jnp.array(something)

def self_sub_fn(a, b):
    # y = x - x should be all zeros; result = y + something
    y = a - a
    return y + b

# Eager reference
ref = jax.block_until_ready(self_sub_fn(x_jax, something_jax))

# JIT compiled — XLA may fold (a - a) into a scalar 0 that broadcasts incorrectly
fn_jit = jax.jit(self_sub_fn)
out = jax.block_until_ready(fn_jit(x_jax, something_jax))

diff = float(jnp.max(jnp.abs(out - ref)))
TOL = 0.01
PASS = diff <= TOL

print(f"max_diff={diff:.6f}  tol={TOL}  pass={PASS}")
print(f"ref shape={ref.shape}  out shape={out.shape}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
