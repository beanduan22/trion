#!/usr/bin/env python3
"""
Bug ID     : jax_xla_mul_zero_elim
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : mul_zero_elim
Root cause : XLA JIT folds x * 0 into zeros BEFORE evaluating x, ignoring NaN/Inf propagation; diverges from eager when x has large values that trigger floating-point edge cases
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

# Values near float32 overflow to trigger fp edge cases
x_np = np.array([3.4e38, -3.4e38, 1.2e38, 0.0, 1.0, -1.0, 3.4e38, -2.1e38],
                dtype=np.float32).reshape(1, 4, 2)
residual_np = np.random.randn(1, 4, 2).astype(np.float32)

x_jax = jnp.array(x_np)
residual_jax = jnp.array(residual_np)

zero = jnp.array(0.0, dtype=jnp.float32)

# Eager reference: (x * 0.0) + residual
ref = jax.block_until_ready((x_jax * zero) + residual_jax)

# JIT compiled — XLA may fold x * 0 to a constant before evaluating x
fn = jax.jit(lambda a, r: (a * 0.0) + r)
out = jax.block_until_ready(fn(x_jax, residual_jax))

diff = float(jnp.max(jnp.abs(jnp.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)
                              - jnp.nan_to_num(ref, nan=0.0, posinf=1e9, neginf=-1e9))))
TOL = 0.01
PASS = diff <= TOL

print(f"max_diff={diff:.6f}  tol={TOL}  pass={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
