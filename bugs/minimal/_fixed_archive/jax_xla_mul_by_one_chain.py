#!/usr/bin/env python3
"""
Bug ID     : jax_xla_mul_by_one_chain
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : mul_by_one_chain
Root cause : XLA JIT folds triple mul-by-one into identity, changing memory layout for downstream conv+tanh
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
try:
    import jax, jax.numpy as jnp, numpy as np
    from jax import lax
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
w = jnp.array(np.random.randn(4, 4, 3, 3).astype(np.float32))
dn = ("NCHW", "OIHW", "NCHW")

def f(x):
    y = x * 1.0 * 1.0 * 1.0
    return jnp.tanh(lax.conv_general_dilated(y, w, (1,1), "SAME", dimension_numbers=dn))

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
