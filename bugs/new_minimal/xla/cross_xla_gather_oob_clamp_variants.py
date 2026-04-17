#!/usr/bin/env python3
"""
Bug ID     : cross_xla_gather_oob_clamp_variants
Source     : Cross-framework testing (2026-04-16) — variant of
             cross_xla_gather_oob_silent_clamp probing the OOB-clamping
             behavior across multiple shapes, axes, and index magnitudes.
Compiler   : TensorFlow XLA 2.21 (jit_compile=True). TF eager / PyTorch /
             ONNX Runtime all surface the OOB; XLA silently clamps.
Patterns   : tf.gather(x, indices=[..., OOB, ...], axis=N)
Root cause : XLA `Gather` lowering applies an implicit
                clamp(idx, 0, dim_size - 1)
             instead of bounds-checking. Holds across:
               - off-by-one (idx = dim_size)         -> dim_size-1
               - far positive (idx >> dim_size)      -> dim_size-1
               - INT_MAX                             -> dim_size-1
               - negative just-OOB (idx = -1)        -> 0
               - negative far-OOB (idx = -dim*10)    -> 0
               - 2D gather, axis=0 or axis=1         -> per-axis clamp
               - mixed valid + multiple OOB indices  -> all OOB -> dim-1
             The clamp direction depends only on sign:
               positive OOB -> upper bound (dim - 1)
               negative OOB -> lower bound (0)

Exit 0 = BUG REPRODUCED on XLA for at least one variant
Exit 1 = not reproduced on any variant
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


def eager(x, idx, axis):
    try:
        return tf.gather(tf.constant(x), indices=idx, axis=axis).numpy().tolist(), None
    except Exception as e:
        return None, type(e).__name__

def torch_select(x, idx, axis):
    try:
        return torch.index_select(torch.tensor(x), axis,
                                  torch.tensor(idx, dtype=torch.long)).numpy().tolist(), None
    except Exception as e:
        return None, type(e).__name__

def xla(x, idx, axis):
    @tf.function(jit_compile=True)
    def f(t):
        return tf.gather(t, indices=idx, axis=axis)
    try:
        return f(tf.constant(x)).numpy().tolist(), None
    except Exception as e:
        return None, type(e).__name__


CASES = [
    ("V1: 1D, off-by-one OOB (idx 256)",
     np.arange(256, dtype=np.float32),  [5, 8, 7, 16, 256, 123], 0),
    ("V2: 1D, far-OOB (idx 1000)",
     np.arange(256, dtype=np.float32),  [5, 8, 1000, 123], 0),
    ("V3: 1D, negative OOB (idx -1)",
     np.arange(256, dtype=np.float32),  [5, -1, 7, 16, 123], 0),
    ("V4: 1D, negative far-OOB (idx -300)",
     np.arange(256, dtype=np.float32),  [5, -300, 7, 123], 0),
    ("V5: 1D, INT_MAX",
     np.arange(256, dtype=np.float32),  [5, 2147483647], 0),
    ("V6: 1D, multiple OOB indices",
     np.arange(256, dtype=np.float32),  [256, 257, 999, 1000000], 0),
    ("V7: 2D, axis=1 OOB on inner dim",
     np.arange(20, dtype=np.float32).reshape(4, 5), [0, 1, 5, 4], 1),
    ("V8: 2D, axis=0 OOB on outer dim",
     np.arange(20, dtype=np.float32).reshape(4, 5), [0, 1, 4, 3], 0),
]

any_bug = False
for label, x, idx, axis in CASES:
    print("=" * 78)
    print(f"{label}")
    print(f"  x.shape={x.shape}  axis={axis}  indices={idx}")
    e_out, e_err = eager(x, idx, axis)
    t_out, t_err = torch_select(x, idx, axis)
    x_out, x_err = xla(x, idx, axis)
    print(f"  TF eager : {'RAISED ' + e_err if e_err else e_out}")
    print(f"  PyTorch  : {'RAISED ' + t_err if t_err else t_out}")
    print(f"  XLA jit  : {'RAISED ' + x_err if x_err else x_out}")
    # Bug = XLA produced output while at least one of the two refs raised.
    if x_out is not None and (e_err or t_err):
        any_bug = True
        print(f"  -> BUG: XLA silently produced output instead of erroring")
    print()

sys.exit(0 if any_bug else 1)
