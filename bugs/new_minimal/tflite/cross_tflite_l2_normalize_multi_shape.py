#!/usr/bin/env python3
"""
Bug ID     : cross_tflite_l2_normalize_multi_shape
Source     : Cross-framework testing (2026-04-16) — variant of
             cross_tflite_l2_normalize_axis_mismatch with two extra shapes
             that exercise the same root cause on rank-3 input and on
             unit-innermost-dim input.
Compiler   : TensorFlow Lite 2.21 — TFLite-only.
             TF eager / XLA / PyTorch eager / ONNX Runtime all return the
             correct whole-tensor L2 normalization.
Patterns   : tf.math.l2_normalize(x)   # no axis arg, default axis=None
Root cause : Same as cross_tflite_l2_normalize_axis_mismatch:
               - tf.math.l2_normalize(x, axis=None) normalizes the whole tensor
               - TFLite's L2_NORMALIZATION op only normalizes the innermost dim
             The converter silently maps the former to the latter.
             Bug manifests for any input where the product of non-innermost
             dims is > 1 (i.e. there is more than one innermost vector).
Test cases :
    Case 3  shape=(2,2,3)  — rank-3, half-step floats 1.0..6.5
    Case 4  shape=(4,1)    — innermost dim = 1, so each scalar normalizes
                              to 1.0 (degenerate visible failure)

Exit 0 = BUG REPRODUCED on TFLite for at least one case
Exit 1 = not reproduced on any case
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


def numpy_ref(x):
    """Whole-tensor L2 normalize, matching tf.math.l2_normalize(axis=None)."""
    return x / np.linalg.norm(x)


def run_keras(x_np):
    class M(tf.keras.Model):
        def __init__(self):
            super(M, self).__init__()
        def call(self, x):
            return tf.math.l2_normalize(x)
    m = M()
    return m(tf.constant(x_np)).numpy(), m


def run_xla(x_np):
    @tf.function(jit_compile=True, input_signature=[tf.TensorSpec(x_np.shape, tf.float32)])
    def f(x):
        return tf.math.l2_normalize(x)
    return f(tf.constant(x_np)).numpy()


def run_torch(x_np):
    xt = torch.tensor(x_np)
    return (xt / torch.sqrt(torch.sum(xt * xt))).numpy()


def run_onnxrt(x_np):
    class TM(torch.nn.Module):
        def forward(self, x):
            return x / torch.sqrt(torch.sum(x * x))
    buf = io.BytesIO()
    torch.onnx.export(TM(), (torch.tensor(x_np),), buf,
                      input_names=["x"], output_names=["y"], opset_version=17)
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_np})[0]


def run_tflite(keras_model, x_np):
    tfl = tf.lite.TFLiteConverter.from_keras_model(keras_model).convert()
    itp = tf.lite.Interpreter(model_content=tfl)
    in_idx = itp.get_input_details()[0]["index"]
    itp.resize_tensor_input(in_idx, list(x_np.shape))
    itp.allocate_tensors()
    itp.set_tensor(in_idx, x_np)
    itp.invoke()
    return itp.get_tensor(itp.get_output_details()[0]["index"])


CASES = [
    ("Case 3: shape=(2,2,3), half-step floats",
     (np.arange(12, dtype=np.float32) * 0.5 + 1.0).reshape(2, 2, 3)),
    ("Case 4: shape=(4,1), unit innermost dim",
     np.array([[2.], [3.], [4.], [5.]], dtype=np.float32)),
]

any_bug = False
for label, X in CASES:
    print("=" * 78)
    print(f"{label}    shape={tuple(X.shape)}")
    print(f"  input    : {X.flatten().tolist()}")
    expected = numpy_ref(X)
    print(f"  expected : {[round(float(v), 4) for v in expected.flatten()]}")

    keras_out, m = run_keras(X)
    xla_out      = run_xla(X)
    torch_out    = run_torch(X)
    ort_out      = run_onnxrt(X)
    tflite_out   = run_tflite(m, X)

    rows = [
        ("Keras eager",  keras_out),
        ("XLA",          xla_out),
        ("PyTorch",      torch_out),
        ("ONNX Runtime", ort_out),
        ("TFLite",       tflite_out),
    ]
    print(f"  {'Runtime':<14} {'Output (rounded 4dp)':<55} {'Match?'}")
    for name, out in rows:
        rounded = [round(float(v), 4) for v in out.flatten()]
        ok = np.allclose(out, expected)
        if name == "TFLite" and not ok:
            any_bug = True
        print(f"  {name:<14} {str(rounded):<55} {'OK' if ok else 'WRONG'}")
    print()

sys.exit(0 if any_bug else 1)
