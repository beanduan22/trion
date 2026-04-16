#!/usr/bin/env python3
"""
Bug ID     : cross_xla_gather_oob_silent_clamp
Source     : Cross-framework testing (2026-04-16)
Compiler   : TensorFlow XLA 2.21 (jit_compile=True). Same XLA backend in
             jax.jit; PyTorch / ORT / TF eager / tf.function-no-XLA all
             surface the OOB.
Patterns   : tf.gather(x, indices=[..., OOB, ...], axis=0)
Root cause : The XLA `Gather` lowering applies an implicit
             `clamp(index, 0, dim_size - 1)` to every gather index instead
             of bounds-checking. An index of `256` on a tensor of length
             256 silently becomes `255` and the corresponding output element
             is the *last* valid value — no exception, no NaN, no warning.
             TF eager (CPU kernel) raises
                InvalidArgumentError: indices[4] = 256 is not in [0, 256)
             tf.function without XLA inherits the same kernel and also
             raises. Once `jit_compile=True` flips lowering to XLA, the
             error vanishes.
             PyTorch (`torch.index_select`) raises `IndexError`; ONNX
             Runtime (`Gather`) raises `INVALID_ARGUMENT`. So XLA is the
             outlier.
Test case  : x = [0, 1, 2, ..., 255]   (length 256)
             indices = [5, 8, 7, 16, 256, 123]
             expected: an exception (index 256 is OOB)
             XLA out : [5, 8, 7, 16, 255, 123]   <-- silently clamped

Exit 0 = BUG REPRODUCED on XLA
Exit 1 = not reproduced (XLA also raises)
Exit 2 = missing deps
"""
import sys

try:
    import numpy as np
    import tensorflow as tf
    import torch
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

try:
    import onnxruntime as ort
    import io
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False

# Diagnostic input: values 0..255 so each gathered element is identifiable.
x_np = np.arange(256, dtype=np.float32)
INDICES = [5, 8, 7, 16, 256, 123]   # 256 is OOB
print(f"x       : range(256), shape={x_np.shape}")
print(f"indices : {INDICES}    (256 is out of range)")
print(f"expected: an exception or NaN at position 4 (index = 256)")
print()

results = []  # (runtime, "raised <Err>" | output, raised?)

# ── TF eager ────────────────────────────────────────────────────────────────
try:
    y = tf.gather(tf.constant(x_np), indices=INDICES, axis=0).numpy()
    results.append(("TF eager", y.tolist(), False))
except Exception as e:
    results.append(("TF eager", f"RAISED {type(e).__name__}", True))

# ── tf.function without XLA ────────────────────────────────────────────────
@tf.function
def f_graph(x):
    return tf.gather(x, indices=INDICES, axis=0)

try:
    y = f_graph(tf.constant(x_np)).numpy()
    results.append(("tf.function", y.tolist(), False))
except Exception as e:
    results.append(("tf.function", f"RAISED {type(e).__name__}", True))

# ── XLA (jit_compile=True) ─────────────────────────────────────────────────
@tf.function(jit_compile=True)
def f_xla(x):
    return tf.gather(x, indices=INDICES, axis=0)

xla_silent = False
try:
    y = f_xla(tf.constant(x_np)).numpy()
    results.append(("XLA jit", y.tolist(), False))
    xla_silent = True
except Exception as e:
    results.append(("XLA jit", f"RAISED {type(e).__name__}", True))

# ── PyTorch ────────────────────────────────────────────────────────────────
try:
    xt = torch.tensor(x_np)
    y = torch.index_select(xt, 0, torch.tensor(INDICES, dtype=torch.long)).numpy()
    results.append(("PyTorch", y.tolist(), False))
except Exception as e:
    results.append(("PyTorch", f"RAISED {type(e).__name__}", True))

# ── ONNX Runtime ───────────────────────────────────────────────────────────
if HAVE_ORT:
    class TM(torch.nn.Module):
        def forward(self, x, idx):
            return torch.index_select(x, 0, idx)
    buf = io.BytesIO()
    torch.onnx.export(
        TM(),
        (torch.tensor(x_np), torch.tensor(INDICES, dtype=torch.long)),
        buf,
        input_names=["x", "idx"],
        output_names=["y"],
        opset_version=17,
    )
    try:
        sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"x": x_np, "idx": np.array(INDICES, dtype=np.int64)})[0]
        results.append(("ONNX Runtime", y.tolist(), False))
    except Exception as e:
        results.append(("ONNX Runtime", f"RAISED {type(e).__name__}", True))

# ── Report ──────────────────────────────────────────────────────────────────
print(f"{'Runtime':<14} {'Result':<60}")
print("-" * 76)
for name, out, raised in results:
    print(f"{name:<14} {str(out):<60}")

# Exit 0 if XLA silently produced output (the bug); 1 otherwise.
if xla_silent:
    print("\nBUG REPRODUCED — XLA silently clamped OOB index 256 -> 255")
    sys.exit(0)
print("\nNOT REPRODUCED — XLA also raised")
sys.exit(1)
