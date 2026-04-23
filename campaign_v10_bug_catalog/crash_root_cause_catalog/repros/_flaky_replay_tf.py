#!/usr/bin/env python3
"""
Clean-process TF+XLA replay driver for ONE crash model. Run once per case
(subprocess.run-style) so GPU state is reset between calls.

Usage:
    python _flaky_replay_tf.py <path-to-crash_NNNNNN.onnx> <gpu_id>

Exit codes:
  0 = crash reproduced (same error signature)
  1 = ran clean (no error)
  2 = different error (escalate for manual look)
  3 = setup error
"""
import os
import sys
import re

if len(sys.argv) < 2:
    print("missing arg"); sys.exit(3)

onnx_path = sys.argv[1]
gpu_id = sys.argv[2] if len(sys.argv) > 2 else "1"

# Isolate GPU BEFORE importing tf
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")

import onnx
import numpy as np

try:
    import tensorflow as tf
    # Limit memory growth so we don't pre-allocate
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception as e:
    print(f"[setup] tf import failed: {e}")
    sys.exit(3)

try:
    import tf2onnx  # noqa
except Exception:
    pass

# Convert ONNX → TF via onnx_tf or tf2onnx reverse? Use tf2onnx.backend is no longer available.
# Easiest: use onnx → onnx_tf → TF graph. But onnx_tf is heavy. Use tf2onnx.convert_from_tf? No.
# Alternative: use onnxruntime-TF-backend? Or load as tf.function via onnx-tf.
try:
    from onnx_tf.backend import prepare as onnx_tf_prepare
    HAVE_ONNX_TF = True
except Exception:
    HAVE_ONNX_TF = False

if not HAVE_ONNX_TF:
    # Fallback: use the xcomp TFBackend which is what the original campaign used.
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(here, "xcomp")):
            sys.path.insert(0, here)
            break
        here = os.path.dirname(here)
    try:
        from xcomp.oracle.tf_backend import TFBackend
    except Exception as e:
        print(f"[setup] no onnx_tf and no xcomp.TFBackend: {e}")
        sys.exit(3)
    backend = TFBackend()
    model = onnx.load(onnx_path)
    inp = model.graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    np.random.seed(0)
    inputs = {inp.name: np.random.randn(*shape).astype(np.float32)}
    # The xcomp TFBackend returns a BackendResult (no exception). Call with
    # optimized=True so we hit the XLA autotuner path that originally crashed.
    result = backend.run(model, inputs, optimized=True)
    if result.ok:
        print(f"RAN OK: out.shape={result.output.shape}")
        sys.exit(1)
    msg = result.error or ""
    if "Autotuning failed" in msg and ("No valid config" in msg or "NOT_FOUND" in msg):
        short = [ln for ln in msg.splitlines() if "Autotuning" in ln]
        print(f"CRASH REPRO (autotune): {(short[0] if short else msg)[:180]}")
        sys.exit(0)
    if "out of memory" in msg.lower():
        print(f"ENV OOM (still noise): {msg.splitlines()[-1][:180]}")
        sys.exit(2)
    print(f"DIFFERENT ERROR: {msg.splitlines()[-1][:220]}")
    sys.exit(2)
else:
    model = onnx.load(onnx_path)
    inp = model.graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    np.random.seed(0)
    x = np.random.randn(*shape).astype(np.float32)
    try:
        tf_rep = onnx_tf_prepare(model)
        out = tf_rep.run({inp.name: x})
        print(f"RAN OK: {out}")
        sys.exit(1)
    except Exception as e:
        msg = str(e)
        if "Autotuning failed" in msg:
            print(f"CRASH REPRO: {msg.splitlines()[0][:200]}")
            sys.exit(0)
        print(f"OTHER ERROR: {msg.splitlines()[-1][:200]}")
        sys.exit(2)
