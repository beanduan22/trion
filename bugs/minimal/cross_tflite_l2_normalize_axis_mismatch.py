#!/usr/bin/env python3
"""
Bug ID     : cross_tflite_l2_normalize_axis_mismatch
Source     : Cross-framework testing (2026-04-16)
Compiler   : TensorFlow Lite 2.21 — TFLite-only.
             TF eager / XLA / PyTorch eager / ONNX Runtime all return the
             correct whole-tensor L2 normalization.
Patterns   : tf.math.l2_normalize(tf.transpose(x))   # no axis arg
Root cause : `tf.math.l2_normalize(x, axis=None)` (the Python default) means
             normalize over the **entire** tensor — sqrt(sum(x**2)) is a
             scalar.  TFLite's built-in `L2_NORMALIZATION` op, however, is
             defined to normalize only along the **innermost** dimension.
             The TFLite converter silently maps the TF op to the TFLite op
             without accounting for this semantic difference.
             ModelAnalyzer dump:
                 Op#0 ADD(...)
                 Op#1 TRANSPOSE(...)
                 Op#2 L2_NORMALIZATION(T#4) -> [T#5]   # innermost-axis only
             For input shape [1,1,2] -> transpose -> [2,1,1] the innermost
             dim has size 1, so every scalar is normalized to 1.0, while the
             true whole-tensor norm yields 1/sqrt(2) ≈ 0.7071.
Test case  : input shape=[1,1,2], all ones; (x+1) -> transpose -> l2_normalize
             expected: [[[0.70711]], [[0.70711]]]
             TFLite : [[[1.0    ]], [[1.0    ]]]

Exit 0 = BUG REPRODUCED on TFLite
Exit 1 = not reproduced (TFLite output matches reference)
Exit 2 = missing deps
"""
import io
import sys

try:
    import numpy as np
    import tensorflow as tf
    import torch
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

# ── Test input ──────────────────────────────────────────────────────────────
INPUT_SHAPE = [1, 1, 2]
X = np.ones(INPUT_SHAPE, dtype=np.float32)


# ── Reference (numpy): whole-tensor L2 normalize ────────────────────────────
def numpy_ref(x):
    y = x + 1.0
    y = np.transpose(y)               # default: reverse axes -> shape [2,1,1]
    return y / np.linalg.norm(y)      # whole-tensor norm

EXPECTED = numpy_ref(X)
print(f"input    : shape={INPUT_SHAPE}, all 1.0")
print(f"expected : {EXPECTED.flatten().tolist()}  (= 1/sqrt(2))")
print()

results = []

# ── TF Eager (Keras) ────────────────────────────────────────────────────────
class M(tf.keras.Model):
    def __init__(self):
        super(M, self).__init__()
    def call(self, x):
        x = tf.add(x, 1)
        return tf.math.l2_normalize(tf.transpose(x))

m = M()
keras_out = m(tf.constant(X)).numpy()
results.append(("Keras eager", keras_out, np.allclose(keras_out, EXPECTED)))

# ── XLA (tf.function jit_compile=True) ─────────────────────────────────────
@tf.function(jit_compile=True, input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)])
def f_xla(x):
    x = tf.add(x, 1)
    return tf.math.l2_normalize(tf.transpose(x))

xla_out = f_xla(tf.constant(X)).numpy()
results.append(("XLA", xla_out, np.allclose(xla_out, EXPECTED)))

# ── PyTorch eager ───────────────────────────────────────────────────────────
xt = torch.tensor(X)
yt = xt + 1.0
yt = yt.permute(*reversed(range(yt.ndim)))   # equivalent to TF default transpose
torch_out = (yt / torch.linalg.vector_norm(yt)).numpy()
results.append(("PyTorch", torch_out, np.allclose(torch_out, EXPECTED)))

# ── ONNX Runtime (export via torch.onnx) ───────────────────────────────────
class TM(torch.nn.Module):
    def forward(self, x):
        y = x + 1.0
        y = y.permute(*reversed(range(y.ndim)))
        return y / torch.sqrt(torch.sum(y * y))   # whole-tensor norm, opset-portable

buf = io.BytesIO()
torch.onnx.export(TM(), (xt,), buf,
                  input_names=["x"], output_names=["y"], opset_version=17)
sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"x": X})[0]
results.append(("ONNX Runtime", ort_out, np.allclose(ort_out, EXPECTED)))

# ── TFLite ──────────────────────────────────────────────────────────────────
tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
itp = tf.lite.Interpreter(model_content=tfl)
itp.allocate_tensors()
in_idx = itp.get_input_details()[0]["index"]
itp.set_tensor(in_idx, X)
itp.invoke()
tflite_out = itp.get_tensor(itp.get_output_details()[0]["index"])
tflite_ok = np.allclose(tflite_out, EXPECTED)
results.append(("TFLite", tflite_out, tflite_ok))

# ── Report ──────────────────────────────────────────────────────────────────
print(f"{'Runtime':<14} {'Output':<40} {'Match?'}")
print("-" * 64)
for name, out, ok in results:
    status = "OK" if ok else "WRONG"
    print(f"{name:<14} {str(out.flatten().tolist()):<40} {status}")

sys.exit(0 if not tflite_ok else 1)
