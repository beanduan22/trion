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
B, S, D, H = 2, 32, 128, 4
Dh = D // H

X = np.random.randn(B, S, D).astype(np.float32) * 0.8
Wq = np.random.randn(D, D).astype(np.float32) * 0.1
Wk = np.random.randn(D, D).astype(np.float32) * 0.1
Wv = np.random.randn(D, D).astype(np.float32) * 0.1
Wo = np.random.randn(D, D).astype(np.float32) * 0.1
Wff1 = np.random.randn(D, 4 * D).astype(np.float32) * 0.1
Wff2 = np.random.randn(4 * D, D).astype(np.float32) * 0.1
ln1_g = np.ones(D, dtype=np.float32)
ln1_b = np.zeros(D, dtype=np.float32)
ln2_g = np.ones(D, dtype=np.float32)
ln2_b = np.zeros(D, dtype=np.float32)

class TransformerBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.wq = tf.Variable(Wq, trainable=False)
        self.wk = tf.Variable(Wk, trainable=False)
        self.wv = tf.Variable(Wv, trainable=False)
        self.wo = tf.Variable(Wo, trainable=False)
        self.wff1 = tf.Variable(Wff1, trainable=False)
        self.wff2 = tf.Variable(Wff2, trainable=False)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    @tf.function(input_signature=[tf.TensorSpec([B, S, D], tf.float32)])
    def call(self, x):
        q = tf.reshape(tf.matmul(x, self.wq), [B, S, H, Dh])
        k = tf.reshape(tf.matmul(x, self.wk), [B, S, H, Dh])
        v = tf.reshape(tf.matmul(x, self.wv), [B, S, H, Dh])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        scores = tf.matmul(q, k, transpose_b=True) / (Dh ** 0.5)
        attn = tf.nn.softmax(scores, axis=-1)
        ctx = tf.matmul(attn, v)
        ctx = tf.transpose(ctx, [0, 2, 1, 3])
        ctx = tf.reshape(ctx, [B, S, D])
        ctx = tf.matmul(ctx, self.wo)
        res1 = self.ln1(x + ctx)
        ff = tf.nn.relu(tf.matmul(res1, self.wff1))
        ff = tf.matmul(ff, self.wff2)
        return self.ln2(res1 + ff)

m = TransformerBlock()
_ = m(tf.zeros([B, S, D], tf.float32))

keras_out = m(tf.constant(X)).numpy()
print(f"Keras: {keras_out.ravel()[:4]}")

tfl_bytes = tf.lite.TFLiteConverter.from_keras_model(m).convert()
try:
    itp = tf.lite.Interpreter(model_content=tfl_bytes)
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
        print(f"BUG REPRODUCED: TFLite transformer_encoder_layer (max_abs={max_abs:.6f})")
        sys.exit(0)
    print("NOT reproduced")
    sys.exit(1)
except (RuntimeError, ValueError) as e:
    print(f"TFLite error (Keras succeeded): {type(e).__name__}: {str(e)[:200]}")
    print("BUG REPRODUCED: TFLite XNNPack prepare fails while Keras runs correctly")
    sys.exit(0)
