import sys

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def main() -> int:
    T, S, K, N, H = 5, 7, 3, 6, 11
    mask = np.r_[:S] <= S // 2

    try:
        jax.config.update("jax_enable_x64", True)
        with jax.debug_infs(True):
            out = jax.nn.dot_product_attention(
                np.zeros((T, N, H), dtype=np.float64),
                np.zeros((S, K, H), dtype=np.float64),
                np.zeros((S, K, H), dtype=np.float64),
                mask=mask,
            )
            out.block_until_ready()
        print("NOT REPRODUCED: no FloatingPointError with debug_infs=True")
        return 1
    except FloatingPointError as e:
        print(f"BUG REPRODUCED: FloatingPointError triggered: {e}")
        return 0
    except Exception as e:
        print(f"NOT REPRODUCED: different exception {type(e).__name__}: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
