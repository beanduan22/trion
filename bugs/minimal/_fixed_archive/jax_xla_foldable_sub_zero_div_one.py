#!/usr/bin/env python3
"""
Bug ID     : jax_xla_foldable_sub_zero_div_one
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : foldable_sub_zero_div_one
Root cause : XLA JIT folds sub-zero and div-one into identities, changing precision tracking for cumsum
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

def f(x):
    return jnp.cumsum((x - 0.0) / 1.0, axis=-1)

def g(x):
    return jnp.cumsum(x, axis=-1)

jitted_f = jax.jit(f)(x)
jitted_g = jax.jit(g)(x)
eager_f = f(x)
diff_jit = float(jnp.max(jnp.abs(jitted_f - jitted_g)))
diff_eager = float(jnp.max(jnp.abs(eager_f - jitted_f)))
diff = max(diff_jit, diff_eager)
print(f"max diff = {diff}")
PASS = diff < 0.01

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
