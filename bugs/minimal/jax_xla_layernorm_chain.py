#!/usr/bin/env python3
"""
Bug ID     : jax_xla_layernorm_chain
Source     : Campaign v1 (fuzzing)
Compiler   : JAX/XLA (jax.jit)
Patterns   : layer_norm_manual_chain
Root cause : XLA JIT fuses layer norm + sin/tanh activation + attention matmul; softmax precision differs
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
wq = jnp.array(np.random.randn(16, 16).astype(np.float32) * 0.1)
wk = jnp.array(np.random.randn(16, 16).astype(np.float32) * 0.1)
wv = jnp.array(np.random.randn(16, 16).astype(np.float32) * 0.1)
g = jnp.ones((16,), dtype=jnp.float32)
b = jnp.zeros((16,), dtype=jnp.float32)

def f(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    ln = (x - mean) / jnp.sqrt(var + 1e-5) * g + b
    h = jnp.sin(jnp.tanh(ln)) * ln
    q, k, v = h @ wq, h @ wk, h @ wv
    attn = jax.nn.softmax(q @ k.swapaxes(-1, -2) * 0.25, axis=-1)
    return attn @ v

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
