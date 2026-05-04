import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def run_case(name: str, params: tuple[int, int, int, int, float]) -> bool:
    b, d1, d2, d3, amp = params
    np.random.seed(123)
    x = np.random.randn(b, d1).astype(np.float32) * 1.5
    a = np.random.randn(d1, d2).astype(np.float32) * amp
    bm = np.random.randn(d2, d3).astype(np.float32) * amp
    w2 = np.random.randn(d3, 32).astype(np.float32) * (amp * 1.7)

    class M(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.a = tf.Variable(a, trainable=False)
            self.b = tf.Variable(bm, trainable=False)
            self.w2 = tf.Variable(w2, trainable=False)

        @tf.function(input_signature=[tf.TensorSpec([b, d1], tf.float32)])
        def call(self, inp):
            p1 = tf.matmul(tf.matmul(inp, self.a), self.b)
            p2 = tf.matmul(inp, tf.matmul(self.a, self.b))
            return tf.matmul(p1 - p2, self.w2)

    m = M()
    _ = m(tf.zeros([b, d1], tf.float32))
    keras_out = m(tf.constant(x)).numpy()
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl_bytes = conv.convert()
    itp = tf.lite.Interpreter(
        model_content=tfl_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    in_idx = itp.get_input_details()[0]["index"]
    itp.resize_tensor_input(in_idx, list(x.shape))
    itp.allocate_tensors()
    out_idx = itp.get_output_details()[0]["index"]
    itp.set_tensor(in_idx, x)
    itp.invoke()
    tfl_out = itp.get_tensor(out_idx)
    max_abs = float(np.abs(keras_out.ravel() - tfl_out.ravel()).max())
    print(f"{name}: max_abs={max_abs}")
    return max_abs > 0.01

def main() -> int:
    bug = False
    bug |= run_case("medium_dims_amp3", (3, 192, 96, 48, 3.0))
    bug |= run_case("small_dims_amp8", (2, 64, 48, 24, 8.0))
    if bug:
        print("BUG REPRODUCED: fp16 quantized associativity break")
        return 0
    print("not reproduced")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
