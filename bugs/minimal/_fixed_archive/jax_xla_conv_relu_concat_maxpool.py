#!/usr/bin/env python3
"""
Bug ID     : jax_xla_conv_relu_concat_maxpool
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : conv_relu_concat_maxpool
Root cause : XLA JIT parallelizes two conv branches and fuses with maxpool; boundary padding differs
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
w1 = jnp.array(np.random.randn(4, 3, 3, 3).astype(np.float32))
w2 = jnp.array(np.random.randn(4, 3, 3, 3).astype(np.float32))
dn = ("NCHW", "OIHW", "NCHW")

def f(x):
    a = jnp.maximum(lax.conv_general_dilated(x, w1, (1,1), "VALID", dimension_numbers=dn), 0)
    b = jnp.maximum(lax.conv_general_dilated(x, w2, (1,1), "SAME", dimension_numbers=dn), 0)
    b = b[:, :, :a.shape[2], :a.shape[3]]
    cat = jnp.concatenate([a, b], axis=1)
    return lax.reduce_window(cat, -jnp.inf, lax.max, (1,1,2,2), (1,1,2,2), "VALID")

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
