#!/usr/bin/env python3
"""
Bug #1830: jax.jit (XLA) produces wrong output vs PyTorch eager reference.

Patterns    : [['branch', 'where_mask_fill'], ['branch', 'glu'], ['branch', 'residual_add_relu'], ['layout', 'gather_layernorm'], ['branch', 'add_relu_sub'], ['broadcast', 'cumsum_last_axis']]
Dependencies: numpy jax
Run         : python unique_1830.py  →  exit 0 = BUG REPRODUCED
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import jax.scipy.special

# ── Model weights ─────────────────────────────────────────────────────────────
_w0 = np.frombuffer(bytes.fromhex("00000000"), dtype=np.float32).reshape([1])
_w1 = np.frombuffer(bytes.fromhex("000080bf"), dtype=np.float32).reshape([1])
_w2 = np.frombuffer(bytes.fromhex("0000004200000042"), dtype=np.float32).reshape([2])
_w3 = np.frombuffer(bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), dtype=np.float32).reshape([1, 32])
_w4 = np.frombuffer(bytes.fromhex("1c29babb31269abcf34285ba0faa01baa01ac7ba466f69bc5754a1bcb5b97ebc8bf69a3bfc2bbebb47a367bcb4298abcea6c6c3c252fe93a8710603cd025ad3b8dd50ebdea58d93ac6040d3c1bf6cd3b80bb013caf0076baece0a0bcf22f1ebd3b66833c2a1ca63c50cffb3b8b44b13b4f5cd63b3281683ca6ec0d3d9bfd7fbcb408ac3b6dbf8fbc2ac9a23b220c55ba7ca0103deca77a3bc5edfabc12e72dbbe3ceff3b683e1a3b8c199b3c22f4fcbc5b41fc3b13014e3be20a92bc479a38bcb3e812bc9435953a2a6426bd1600023b66b806bd275dfcba56fc19bdb384a33cb2da053cd0c5b63bc5d79d3ca1382f3c227fb83c2e62493c504da2bb22246cbc20f3043d10f810bcd55da23c97b3ee3cac1f96b9f3d51bbb8bb7c7bc7b03fabc3b15c43c3e8698bc229316bbae4464bcb3bbe3ba68de843bb3b0ae3b77ac68bc00bca0bc9531dc3c4efe60bc408ead3c0f7442bb26f0e13c454b0dbdeab0aebb50fce5b957a6833cab85c4ba555a6ebcbab68a3cd7c7463c662f303cb6e78b3c21b905bd9789753c6b5c6cbcde9eadbb385f46bc6358113d0253623c20cbb93c90432eba6d48b73bae76ad3c7d8d3ebcd3c1563d26c9253cce52a63b791fc13c14a0e43b5469a7bccd8d04bb70977dbc1bb3b63c6435a53a934137bdc80f06bd5543733b090bffbbef0749bb094f36bdb395b1bcfdfed13cc63581bb65fa35bc"), dtype=np.float32).reshape([4, 32])
_w5 = np.frombuffer(bytes.fromhex("00000000"), dtype=np.float32).reshape([1])
_w6 = np.frombuffer(bytes.fromhex("0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f"), dtype=np.float32).reshape([32])
_w7 = np.frombuffer(bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), dtype=np.float32).reshape([32])
_w8 = np.frombuffer(bytes.fromhex("0000803f"), dtype=np.float32).reshape([1])

# ── Triggering input ─────────────────────────────────────────────────────────
INPUT = np.frombuffer(bytes.fromhex("277ab03d1f22f1bdc7b0d43c78a4d7bd271865bc321058bce21a1ebea3a1ae3d684b6cbdeb50adbdbf2475bdcd1c6abd301ca3bd913525be701731bd5af7713c6f540d3d33a7fc3dd3bc723c3b72bf3d1bcb79bdf12a12bd2da10fbef660a8bd6a53e63c088155be3425403d2b16caba8785143ea5a8373d184a953eea46e03d77933abdff40a9bd8f12303e25769e3dc853eebddad4d9bd8b53b2bceddc0bbeb3c7e1bd616e8e3bda5283bdd7e0663e0a2decbd57cb80bc4cbc373d2f332d3e096e02beb2ede0bc4068eb3db76618beaf3e273e135721bc1c6ba23c5644a7bc6b2a13be62ad0b3e8fc5283e77eb95bd844822bea0c6853caac51b3ef7919a3d"), dtype=np.float32).reshape([1, 64])  # input "model_input"

EXPECTED = np.frombuffer(bytes.fromhex("16b5c841681ac241458fbd415e42b741413fb041de97ab4196a0a34136199b417909924156558f41794a8c41115e8a41a35c87412b927e416c15614170995b41670c5241e79434418d072d41fb25254159671e41da281741bd6f1341775106410a09f740cf0eef40a0748940508d734026fc4e40a8e6414020662f402fbb883f"), dtype=np.float32)  # pytorch_eager reference


# ── Model computation (translated from ONNX) ──────────────────────────────────
def model(x):
    n0_wmf_gt = (x > _w0)
    n0_wmf_wh = jnp.where(n0_wmf_gt, x, _w1)
    n0_wmf_out = n0_wmf_wh * x
    _split_3 = jnp.split(n0_wmf_out, [32], axis=-1)
    n100_glu_a = _split_3[0]
    n100_glu_b = _split_3[1]
    n100_glu_sig = (np.float32(1.0) / (np.float32(1.0) + jnp.exp(-n100_glu_b)))
    n100_glu_out = n100_glu_a * n100_glu_sig
    n200_rarelu_add = n100_glu_out + _w3
    n200_rarelu_out = jnp.maximum(n200_rarelu_add, np.float32(0.0))
    n300_gln_gth = jnp.take(_w4, _w5.astype(np.int32), axis=0)
    n300_gln_added = n300_gln_gth + n200_rarelu_out
    n300_gln_out = (((lambda _x: (lambda _m, _v: (_x - _m) / jnp.sqrt(_v + np.float32(9.999999747378752e-06)))(jnp.mean(_x, axis=tuple(range(-1%_x.ndim, _x.ndim)), keepdims=True), jnp.mean((_x - jnp.mean(_x, axis=tuple(range(-1%_x.ndim, _x.ndim)), keepdims=True))**2, axis=tuple(range(-1%_x.ndim, _x.ndim)), keepdims=True)))(n300_gln_added)) * _w6) + _w7
    n400_ars_ad = n300_gln_out + n300_gln_out
    n400_ars_rl = jnp.maximum(n400_ars_ad, np.float32(0.0))
    n400_ars_out = n400_ars_rl - n300_gln_out
    n500_cs_out = jnp.cumsum(n400_ars_out, axis=int(_w8.flat[0]))
    return jnp.asarray(n500_cs_out, dtype=jnp.float32)


if __name__ == "__main__":
    x_jax = jnp.array(INPUT)
    out = np.array(jax.jit(model)(x_jax), dtype=np.float32).ravel()

    diff = float(np.linalg.norm(EXPECTED.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(EXPECTED.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {EXPECTED[:6]}")
    print(f"actual   (jax.jit/XLA) : {out[:6]}")
    print(f"rel L2 : {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)
