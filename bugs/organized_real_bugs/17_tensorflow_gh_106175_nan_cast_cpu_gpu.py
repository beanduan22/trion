import sys

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def main() -> int:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    x = tf.constant([np.nan], dtype=tf.float32)
    with tf.device("/CPU:0"):
        cpu = int(tf.cast(x, tf.int32).numpy()[0])
    gpus = tf.config.list_physical_devices("GPU")
    print("cpu:", cpu)
    print("gpus:", gpus)
    if not gpus:
        return 2
    with tf.device("/GPU:0"):
        gpu = int(tf.cast(x, tf.int32).numpy()[0])
    print("gpu:", gpu)
    if cpu != gpu:
        print("BUG REPRODUCED: CPU and GPU differ for tf.cast(NaN, int32)")
        return 0
    print("NOT REPRODUCED: CPU and GPU agree")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
