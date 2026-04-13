#!/usr/bin/env python3
"""
Bug ID     : jax_xla_dilated_branch_sum
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : dilated_branch_sum
Root cause : XLA JIT fuses multi-dilation branch sum, possibly reordering or vectorizing differently
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
x = jnp.array(np.random.randn(1, 4, 16, 16).astype(np.float32))
w = jnp.array(np.random.randn(4, 4, 3, 3).astype(np.float32))
dn = ("NCHW", "OIHW", "NCHW")

def f(x):
    a = lax.conv_general_dilated(x, w, (1,1), "SAME", rhs_dilation=(1,1), dimension_numbers=dn)
    b = lax.conv_general_dilated(x, w, (1,1), "SAME", rhs_dilation=(2,2), dimension_numbers=dn)
    c = lax.conv_general_dilated(x, w, (1,1), "SAME", rhs_dilation=(4,4), dimension_numbers=dn)
    return a + b + c

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
