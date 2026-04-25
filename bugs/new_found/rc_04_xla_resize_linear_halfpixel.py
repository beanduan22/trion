#!/usr/bin/env python3
"""
Root Cause 4 — XLA / JAX disagrees with ONNX on Resize(linear, half_pixel).

Covers bugs : 91, 136 (and contributes drift to several other XLA divergences)
Backends    : onnxruntime (reference) vs xla (jax.image.resize)

Graph: a single Resize node, NCHW float32, linear interpolation,
half_pixel coordinate transform — the most common Resize configuration.

The minimal reproducer shows that the divergence is direction-dependent:
* upsampling   (30→32, 64→96): rel_L2 ≈ 3e-6   (agree)
* downsampling (32→30, 32→15): rel_L2 ≈ 0.10–0.70   (large mismatch)

Root cause: ORT implements the ONNX-spec linear-with-half-pixel kernel,
which does NOT antialias on downsample (each output pixel is a 2×2 bilinear
read from the input). `jax.image.resize(method="linear")` automatically
applies an anti-alias prefilter when the output is smaller than the input —
the result is mathematically reasonable but does not match ONNX semantics,
so any model that downsamples through Resize is silently wrong on XLA.

Usage: cd Xcomp_V2 && PYTHONPATH=. python rc_04_xla_resize_linear_halfpixel.py
Exit 0 = bug reproduced, 1 = not reproduced.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

warnings.filterwarnings("ignore")


def build_resize(Hin: int, Win: int, Hout: int, Wout: int,
                 N: int = 1, C: int = 3) -> onnx.ModelProto:
    x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, Hin, Win])
    y_vi = helper.make_tensor_value_info("y", TensorProto.FLOAT, [N, C, Hout, Wout])
    roi    = numpy_helper.from_array(np.array([], np.float32), "roi")
    scales = numpy_helper.from_array(np.array([], np.float32), "scales")
    sizes  = numpy_helper.from_array(np.array([N, C, Hout, Wout], np.int64), "sizes")
    nodes = [helper.make_node("Resize",
                              ["x", "roi", "scales", "sizes"], ["y"],
                              mode="linear",
                              coordinate_transformation_mode="half_pixel")]
    g = helper.make_graph(nodes, "rs_lin_hp", [x_vi], [y_vi], [roi, scales, sizes])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def run_pair(Hin: int, Win: int, Hout: int, Wout: int):
    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
    from trion.oracle.xla_backend import XLABackend
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1, 3, Hin, Win)).astype(np.float32)
    m = build_resize(Hin, Win, Hout, Wout)
    r1 = ONNXRuntimeBackend().run(m, {"x": x}, optimized=True)
    r2 = XLABackend().run(m, {"x": x}, optimized=True)
    if not (r1.ok and r2.ok):
        return None
    ref = np.asarray(r1.output, np.float64).ravel()
    out = np.asarray(r2.output, np.float64).ravel()
    rel = float(np.linalg.norm(ref - out) / (np.linalg.norm(ref) + 1e-8))
    ad  = float(np.max(np.abs(ref - out)))
    return ref, out, rel, ad


def main() -> int:
    print("Direction          rel_L2     max_abs   ORT[0]      XLA[0]")
    print("-" * 60)
    cases = [
        ("upsample 30→32",   (30, 30, 32, 32)),
        ("upsample 64→96",   (64, 64, 96, 96)),
        ("downsample 32→30", (32, 32, 30, 30)),
        ("downsample 32→15", (32, 32, 15, 15)),
    ]
    bug_lines = []
    for label, (Hi, Wi, Ho, Wo) in cases:
        r = run_pair(Hi, Wi, Ho, Wo)
        if r is None:
            print(f"{label:18s} <run failed>"); continue
        ref, out, rel, ad = r
        flag = " <-- BUG" if rel > 0.05 else ""
        print(f"{label:18s} {rel:9.6f}  {ad:9.3e}  {ref[0]:+9.4f}  {out[0]:+9.4f}{flag}")
        if rel > 0.05:
            bug_lines.append((label, rel, ad))

    if bug_lines:
        print("\nBUG REPRODUCED — XLA's Resize does not match ONNX spec when downsampling.")
        print(f"Cases above tolerance: {len(bug_lines)}/{len(cases)}.")
        worst = max(bug_lines, key=lambda r: r[1])
        print(f"Worst: {worst[0]}  rel_L2={worst[1]:.4f}  max_abs={worst[2]:.3f}.")
        print("Models that downsample at any stage will silently disagree on XLA.")
        return 0
    print("not reproduced")
    return 1


if __name__ == "__main__":
    sys.exit(main())
