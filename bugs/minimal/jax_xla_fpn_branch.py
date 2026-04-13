#!/usr/bin/env python3
"""
Bug ID     : jax_xla_fpn_branch
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : fpn_branch
Root cause : XLA JIT fuses FPN upsample+skip-add+conv+tanh pipeline, causing FP divergence at boundaries
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
x = jnp.array(np.random.randn(1, 4, 4, 4).astype(np.float32))
skip = jnp.array(np.random.randn(1, 4, 8, 8).astype(np.float32))
w1 = jnp.array(np.random.randn(4, 4, 1, 1).astype(np.float32))
w2 = jnp.array(np.random.randn(4, 4, 1, 1).astype(np.float32))
dn = ("NCHW", "OIHW", "NCHW")

def f(x):
    c = lax.conv_general_dilated(x, w1, (1,1), "SAME", dimension_numbers=dn)
    up = jax.image.resize(c, (1, 4, 8, 8), method="nearest")
    s = up + skip
    out = lax.conv_general_dilated(s, w2, (1,1), "SAME", dimension_numbers=dn)
    return jnp.tanh(out)

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
