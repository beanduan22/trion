import sys

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def run_case(name: str, values: list[float]) -> bool:
    x = tf.constant(values, dtype=tf.float32)
    with tf.device("/CPU:0"):
        cpu = tf.cast(x, tf.int32).numpy().tolist()
    with tf.device("/GPU:0"):
        gpu = tf.cast(x, tf.int32).numpy().tolist()
    print(f"{name}:")
    print(f"  cpu={cpu}")
    print(f"  gpu={gpu}")
    return cpu != gpu

def main() -> int:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if not tf.config.list_physical_devices("GPU"):
        print("missing gpu")
        return 2
    bug = False
    bug |= run_case("two_nans_float32_to_int32", [np.nan, np.nan])
    bug |= run_case("nan_mixed_with_finite_float32_to_int32", [np.nan, 7.25, -3.5])
    if bug:
        print("BUG REPRODUCED: CPU/GPU NaN cast inconsistency")
        return 0
    print("not reproduced")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
