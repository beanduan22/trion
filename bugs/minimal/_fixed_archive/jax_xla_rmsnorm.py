#!/usr/bin/env python3
"""
Bug ID     : jax_xla_rmsnorm
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : manual_rmsnorm_last_axis
Root cause : XLA JIT fuses RMS norm (sqrt(mean(x^2)+eps)) with gated MLP, numerical divergence in sqrt
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
try:
    import jax, jax.numpy as jnp, numpy as np
except ImportError:
    print("missing deps"); import sys; sys.exit(2)

try:
    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError
except Exception:
    print("No JAX GPU backend; skipping (GPU-only bug)")
    PASS = True
    import sys as _sys
    print("not reproduced"); _sys.exit(1)

np.random.seed(42)
x = jnp.array(np.random.randn(1, 4, 16).astype(np.float32))
w1 = jnp.array(np.random.randn(16, 16).astype(np.float32) * 0.1)
w2 = jnp.array(np.random.randn(16, 16).astype(np.float32) * 0.1)

def f(x):
    rms = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    gate = jax.nn.sigmoid(rms @ w1)
    return gate * (rms @ w2)

eager = f(x)
jitted = jax.jit(f)(x)
diff = float(jnp.max(jnp.abs(eager - jitted)))
print(f"max diff = {diff}")
PASS = diff < 0.01

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
