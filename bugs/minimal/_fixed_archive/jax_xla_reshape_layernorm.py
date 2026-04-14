#!/usr/bin/env python3
"""
Bug ID     : jax_xla_reshape_layernorm
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : reshape_layernorm_reshape
Root cause : XLA JIT eliminates reshape-norm-reshape and identity transpose; fused path computes differently
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
x = jnp.array(np.random.randn(1, 4, 8, 8).astype(np.float32))
g = jnp.array(np.random.randn(64).astype(np.float32))
b = jnp.array(np.random.randn(64).astype(np.float32))

def f(x):
    flat = x.reshape(4, 64)
    mean = jnp.mean(flat, axis=-1, keepdims=True)
    var = jnp.var(flat, axis=-1, keepdims=True)
    ln = (flat - mean) / jnp.sqrt(var + 1e-5) * g + b
    return ln.reshape(1, 4, 8, 8)

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
