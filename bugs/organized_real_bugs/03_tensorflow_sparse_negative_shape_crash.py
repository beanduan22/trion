import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def run_case(name: str, indices, values, b_cols: int) -> None:
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    sp_a_indices = np.array(indices, dtype=np.int64)
    sp_a_values = np.array(values, dtype=np.float32)
    sp_a_shape = np.array([-1, 3], dtype=np.int64)
    b = np.arange(3 * b_cols, dtype=np.float32).reshape(3, b_cols)
    sp_a = tf.compat.v1.SparseTensor(indices=sp_a_indices, values=sp_a_values, dense_shape=sp_a_shape)
    out = tf.compat.v1.sparse.sparse_dense_matmul(sp_a, b)
    with tf.compat.v1.Session() as sess:
        print(f"running {name}")
        print(sess.run(out))

def main() -> int:
    print("case=same_shape_more_cols_output")
    run_case("same_shape_more_cols_output", [[0, 0], [1, 1], [2, 2]], [1.0, 2.0, 3.0], 4)
    print("not reproduced")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
