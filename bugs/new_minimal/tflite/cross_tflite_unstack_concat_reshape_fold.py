#!/usr/bin/env python3
"""
Bug ID     : cross_tflite_unstack_concat_reshape_fold
Source     : Cross-framework testing (2026-04-16)
Compiler   : TensorFlow Lite 2.21 — TFLite-only.
             TF eager / XLA / PyTorch eager / ONNX Runtime all return the
             correct (transposed) result.
Patterns   : reshape(x, [H, W, 1]) -> unstack(axis=1) -> concat(axis=0) -> reshape(x, [H, W])
Root cause : The TFLite converter's pattern-rewriter folds the entire chain
             into a single no-op RESHAPE [H,W] -> [H,W], ignoring the fact
             that `unstack(axis=1) + concat(axis=0)` permutes elements
             (transpose-equivalent on the [H,W] endpoint).
             ModelAnalyzer dump confirms the converted subgraph contains
             only `Op#0 RESHAPE(T#0, [H, W]) -> T#2`.
             TF eager, XLA (jit_compile=True), PyTorch eager, and ONNX
             Runtime (via torch.onnx, opset 17) all preserve the
             permutation correctly.
Test case  : shape=(4, 3), values 1..12
             expected: [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]
             TFLite : [1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12]  (input unchanged)

Exit 0 = BUG REPRODUCED on TFLite
Exit 1 = not reproduced (TFLite output matches reference)
Exit 2 = missing deps
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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

# ── Test input: shape=(4, 3), values 1..12 ─────────────────────────────────
X = np.array(
    [[1., 2., 3.],
     [4., 5., 6.],
     [7., 8., 9.],
     [10., 11., 12.]],
    dtype=np.float32,
)
H, W = X.shape

# Reference (numpy): trace the op chain explicitly
def numpy_ref(x):
    a = x.reshape(H, W, 1)
    b = np.concatenate(list(np.split(a, W, axis=1)), axis=0)
    return b.reshape(H, W)

EXPECTED = numpy_ref(X)
print(f"input    : {X.flatten().tolist()}")
print(f"expected : {EXPECTED.flatten().tolist()}")
print()

results = []  # (runtime, output, ok)

# ── TF Eager (Keras) ────────────────────────────────────────────────────────
class M(tf.keras.Model):
    def __init__(self):
        super(M, self).__init__()
    @tf.function(input_signature=[tf.TensorSpec([H, W], tf.float32)])
    def call(self, x):
        a = tf.reshape(x, [H, W, 1])
        b = tf.concat(tf.unstack(a, axis=1), 0)
        return tf.reshape(b, [H, W])

m = M()
keras_out = m(tf.constant(X)).numpy()
results.append(("Keras eager", keras_out, np.allclose(keras_out, EXPECTED)))

# ── XLA (tf.function jit_compile=True) ─────────────────────────────────────
@tf.function(jit_compile=True, input_signature=[tf.TensorSpec([H, W], tf.float32)])
def f_xla(x):
    a = tf.reshape(x, [H, W, 1])
    b = tf.concat(tf.unstack(a, axis=1), 0)
    return tf.reshape(b, [H, W])

xla_out = f_xla(tf.constant(X)).numpy()
results.append(("XLA", xla_out, np.allclose(xla_out, EXPECTED)))

# ── PyTorch eager ───────────────────────────────────────────────────────────
xt = torch.tensor(X)
a = xt.reshape(H, W, 1)
b = torch.cat(list(torch.unbind(a, dim=1)), dim=0)
torch_out = b.reshape(H, W).numpy()
results.append(("PyTorch", torch_out, np.allclose(torch_out, EXPECTED)))

# ── ONNX Runtime (export via torch.onnx) ───────────────────────────────────
class TM(torch.nn.Module):
    def forward(self, x):
        a = x.reshape(H, W, 1)
        b = torch.cat(list(torch.unbind(a, dim=1)), dim=0)
        return b.reshape(H, W)

buf = io.BytesIO()
torch.onnx.export(TM(), (xt,), buf, input_names=["x"], output_names=["y"], opset_version=17)
sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"x": X})[0]
results.append(("ONNX Runtime", ort_out, np.allclose(ort_out, EXPECTED)))

# ── TFLite ──────────────────────────────────────────────────────────────────
tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
itp = tf.lite.Interpreter(
    model_content=tfl,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
in_idx = itp.get_input_details()[0]["index"]
itp.resize_tensor_input(in_idx, [H, W])
itp.allocate_tensors()
itp.set_tensor(in_idx, X)
itp.invoke()
tflite_out = itp.get_tensor(itp.get_output_details()[0]["index"])
tflite_ok = np.allclose(tflite_out, EXPECTED)
results.append(("TFLite", tflite_out, tflite_ok))

# ── Report ──────────────────────────────────────────────────────────────────
print(f"{'Runtime':<14} {'Output':<60} {'Match?'}")
print("-" * 86)
for name, out, ok in results:
    status = "OK" if ok else "WRONG"
    print(f"{name:<14} {str(out.flatten().tolist()):<60} {status}")

# Exit code: 0 if TFLite reproduces the bug, 1 otherwise
sys.exit(0 if not tflite_ok else 1)
