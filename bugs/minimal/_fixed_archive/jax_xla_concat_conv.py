#!/usr/bin/env python3
"""
Bug ID     : jax_xla_concat_conv
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : concat_conv
Root cause : XLA JIT fuses concat+conv, may reorder channel reduction causing different FP accumulation
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
x = jnp.array(np.random.randn(1, 3, 8, 8).astype(np.float32))
const = jnp.array(np.random.randn(1, 5, 8, 8).astype(np.float32))
w = jnp.array(np.random.randn(4, 8, 1, 1).astype(np.float32))

def f(x):
    cat = jnp.concatenate([x, const], axis=1)
    return lax.conv_general_dilated(cat, w, (1, 1), "SAME",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))

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
