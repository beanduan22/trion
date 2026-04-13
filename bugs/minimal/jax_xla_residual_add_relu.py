#!/usr/bin/env python3
"""
Bug ID     : jax_xla_residual_add_relu
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : residual_add_relu
Root cause : XLA JIT fuses residual-add-relu with erf/tanh chain; erf(tanh(x))*x computation diverges
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
try:
    import jax, jax.numpy as jnp, jax.scipy.special, numpy as np
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
bias = jnp.array(np.random.randn(1, 4, 1, 1).astype(np.float32))

def f(x):
    relu = jnp.maximum(x + bias, 0.0)
    return jnp.abs(-jax.scipy.special.erf(jnp.tanh(relu)) * relu)

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
