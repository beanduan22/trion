import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

np.random.seed(42)
B, S, D = 2, 128, 64
H = 4  # heads
assert D % H == 0
Dh = D // H

X = np.random.randn(B, S, D).astype(np.float32) * 2.0
Wq = np.random.randn(D, D).astype(np.float32) * 1.0
Wk = np.random.randn(D, D).astype(np.float32) * 1.0
Wv = np.random.randn(D, D).astype(np.float32) * 1.0
CAP = 5.0

class AttnSoftcap(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.wq = tf.Variable(Wq, trainable=False)
        self.wk = tf.Variable(Wk, trainable=False)
        self.wv = tf.Variable(Wv, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([B, S, D], tf.float32)])
    def call(self, x):
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)
        q = tf.reshape(q, [B, S, H, Dh])
        k = tf.reshape(k, [B, S, H, Dh])
        v = tf.reshape(v, [B, S, H, Dh])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        scores = tf.matmul(q, k, transpose_b=True) / (Dh ** 0.5)
        scores = tf.tanh(scores / CAP) * CAP
        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, S, D])
        return out

m = AttnSoftcap()
_ = m(tf.zeros([B, S, D], tf.float32))  # build

keras_out = m(tf.constant(X)).numpy()
print(f"Keras: {keras_out.ravel()[:4]}")

conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl_bytes = conv.convert()
try:
    itp = tf.lite.Interpreter(
        model_content=tfl_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    in_idx = itp.get_input_details()[0]["index"]
    itp.resize_tensor_input(in_idx, list(X.shape))
    itp.allocate_tensors()
    out_idx = itp.get_output_details()[0]["index"]
    itp.set_tensor(in_idx, X)
    itp.invoke()
    tfl_out = itp.get_tensor(out_idx)
    print(f"TFLite:{tfl_out.ravel()[:4]}")
    max_abs = float(np.abs(keras_out.ravel() - tfl_out.ravel()).max())
    print(f"max_abs={max_abs:.6f}")
    if max_abs > 0.01:
        print(f"BUG REPRODUCED: TFLite attention_logit_softcap (max_abs={max_abs:.6f})")
        sys.exit(0)
    print("NOT reproduced")
    sys.exit(1)
except (RuntimeError, ValueError) as e:
    print(f"TFLite error (Keras succeeded): {type(e).__name__}: {str(e)[:200]}")
    print("BUG REPRODUCED: TFLite XNNPack prepare fails while Keras runs correctly")
    sys.exit(0)
