#!/usr/bin/env python3
"""
Bug ID     : jax_xla_identity_chain
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : identity_chain_two
Root cause : XLA JIT optimizes away identity but changes numerical path for floor/ceil chain
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
x = jnp.array(np.random.randn(1, 1, 8, 8).astype(np.float32))
w = jnp.array(np.random.randn(8, 8).astype(np.float32))

def f(x):
    y = x + 0.0  # identity
    z = jnp.ceil(jnp.floor(y) * y)
    flat = z.reshape(z.shape[0], -1)
    return jnp.matmul(flat, w)

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
