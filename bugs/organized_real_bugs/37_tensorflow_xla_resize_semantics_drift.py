import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # force CPU

import sys

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

TOL = 1e-4

def compare(name, eager, xla):
    e = np.asarray(eager).ravel()
    x = np.asarray(xla).ravel()
    diff = float(np.max(np.abs(e - x)))
    print(f"[{name}]")
    print(f"  eager first 8 : {e[:8]}")
    print(f"  xla   first 8 : {x[:8]}")
    print(f"  max_abs diff  : {diff:.6g}")
    return diff

def run_case(name, builder, input_tensor):
    eager_fn = tf.function(builder(), jit_compile=False)
    xla_fn = tf.function(builder(), jit_compile=True)
    eager_out = eager_fn(input_tensor).numpy()
    xla_out = xla_fn(input_tensor).numpy()
    return compare(name, eager_out, xla_out)

def mk_resize(method, new_h, new_w, antialias=False):
    def build():
        def f(t):
            return tf.image.resize(t, [new_h, new_w], method=method, antialias=antialias)
        return f
    return build

rng = np.random.RandomState(0)
cases = []

for (H, W, NH, NW) in [(3, 3, 5, 5), (5, 5, 8, 8), (4, 4, 7, 7), (3, 5, 5, 8), (7, 7, 3, 3)]:
    x = tf.constant(np.arange(1, H * W + 1).reshape(1, H, W, 1).astype(np.float32))
    cases.append((f"nearest {H}x{W}->{NH}x{NW}", mk_resize("nearest", NH, NW), x))

for (H, W, NH, NW) in [(8, 8, 3, 3), (16, 16, 5, 5), (7, 7, 3, 3)]:
    x = tf.constant(rng.randn(1, H, W, 1).astype(np.float32))
    cases.append((
        f"bilinear-antialias {H}x{W}->{NH}x{NW}",
        mk_resize("bilinear", NH, NW, antialias=True),
        x,
    ))

for (H, W, NH, NW) in [(5, 5, 8, 8), (7, 7, 11, 11)]:
    x = tf.constant(rng.randn(1, H, W, 1).astype(np.float32))
    cases.append((f"bicubic {H}x{W}->{NH}x{NW}", mk_resize("bicubic", NH, NW), x))

def conv_builder():
    W = tf.constant(rng.randn(3, 3, 1, 2).astype(np.float32))

    def f(t):
        return tf.nn.conv2d(t, W, strides=[1, 2, 2, 1], padding="SAME")

    return f

cx = tf.constant(rng.randn(1, 7, 7, 1).astype(np.float32))
cases.append(("conv2d 7x7 stride2 SAME", conv_builder, cx))

divergences = []
for name, builder, input_tensor in cases:
    try:
        diff = run_case(name, builder, input_tensor)
        if diff > TOL:
            divergences.append((name, diff))
    except Exception as e:
        print(f"[{name}] error: {type(e).__name__}: {str(e)[:120]}")

print()
if divergences:
    print(f"BUG REPRODUCED — {len(divergences)} case(s) diverge between XLA and eager:")
    for n, d in divergences:
        print(f"  - {n}: max_abs diff = {d:.6g}")
    sys.exit(0)

print("not reproduced — XLA and eager agree within tolerance on all cases")
sys.exit(1)
