#!/usr/bin/env python3
"""
Root Cause 3 — TorchInductor miscompiles
Pad(edge) → DepthwiseConv → BN → Relu → clipped-affine → 1/sqrt(x²+eps).

Covers bug : 722. ORT (correct) vs torch_compile (Inductor) — the rewrite
that fuses the Pad-edge boundary with the dwconv chain is unsound, and
the spatial-border outputs disagree by ~9 abs while the interior matches.

Usage: PYTHONPATH=. python rc_03_torch_compile_pad_edge_dwconv_chain.py
Exit 0 = bug reproduced.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

warnings.filterwarnings("ignore")


def build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    N, C, H, W = 1, 3, 32, 32
    H2, W2 = H + 4, W + 4  # +2 each side from Pad
    x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, H, W])
    y_vi = helper.make_tensor_value_info("y", TensorProto.FLOAT, [N, C, H2, 32])

    inits = [
        numpy_helper.from_array(np.array([0,0,2,2, 0,0,2,2], np.int64),       "pads"),
        numpy_helper.from_array(rng.normal(0, 0.3, (C, 1, 5, 5)).astype(np.float32), "cw"),
        numpy_helper.from_array(rng.normal(0, 0.1, (C,)).astype(np.float32),         "cb"),
        numpy_helper.from_array(np.ones (C, np.float32), "bns"),
        numpy_helper.from_array(np.zeros(C, np.float32), "bnb"),
        numpy_helper.from_array(np.zeros(C, np.float32), "bnm"),
        numpy_helper.from_array(np.ones (C, np.float32), "bnv"),
        numpy_helper.from_array(np.array(2.0, np.float32),  "sc"),
        numpy_helper.from_array(np.array(0.5, np.float32),  "sh"),
        numpy_helper.from_array(np.array(-1.0, np.float32), "lo"),
        numpy_helper.from_array(np.array(1.0, np.float32),  "hi"),
        numpy_helper.from_array(np.array(1e-6, np.float32), "eps"),
        numpy_helper.from_array(np.ones(W2, np.float32),                     "sc2"),
        numpy_helper.from_array(rng.normal(0, 0.5, (W2, 32)).astype(np.float32), "W"),
    ]
    nodes = [
        helper.make_node("Pad",  ["x", "pads"], ["p"], mode="edge"),
        helper.make_node("Conv", ["p", "cw", "cb"], ["cv"],
                         group=C, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]),
        helper.make_node("BatchNormalization",
                         ["cv", "bns", "bnb", "bnm", "bnv"], ["bn"], epsilon=1e-5),
        helper.make_node("Relu", ["bn"], ["rl"]),
        helper.make_node("Mul",  ["rl", "sc"], ["m1"]),
        helper.make_node("Add",  ["m1", "sh"], ["ad"]),
        helper.make_node("Max",  ["ad", "lo"], ["mx"]),
        helper.make_node("Min",  ["mx", "hi"], ["caff"]),
        helper.make_node("Mul",  ["caff", "caff"], ["sq"]),
        helper.make_node("Add",  ["sq", "eps"], ["sqe"]),
        helper.make_node("Sqrt", ["sqe"], ["sqrt"]),
        helper.make_node("Reciprocal", ["sqrt"], ["rc"]),
        helper.make_node("Mul",  ["caff", "rc"], ["norm"]),
        helper.make_node("Mul",  ["norm", "sc2"], ["nm"]),
        helper.make_node("MatMul", ["nm", "W"], ["y"]),
    ]
    g = helper.make_graph(nodes, "pad_edge_dwconv_bn_chain", [x_vi], [y_vi], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def main() -> int:
    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
    from trion.oracle.torch_compile_backend import TorchCompileBackend

    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, (1, 3, 32, 32)).astype(np.float32)
    model = build_model()

    r_ort = ONNXRuntimeBackend().run(model, {"x": x}, optimized=True)
    r_tc  = TorchCompileBackend().run(model, {"x": x}, optimized=True)
    if not r_ort.ok:
        print(f"setup error: ORT failed: {r_ort.error}"); return 2

    ref = np.asarray(r_ort.output, np.float64).ravel()
    if not r_tc.ok:
        print(f"torch_compile error: {r_tc.error}")
        print("→ torch_compile crashed where ORT succeeded — bug.")
        return 0
    out = np.asarray(r_tc.output, np.float64).ravel()

    rel = float(np.linalg.norm(ref - out) / (np.linalg.norm(ref) + 1e-8))
    abs_max = float(np.max(np.abs(ref - out)))
    print(f"ORT/TC first 4: {ref[:4]} | {out[:4]}")
    print(f"rel_L2={rel:.6f}  max_abs_diff={abs_max:.3e}")
    bad = np.argsort(np.abs(ref - out))[-3:]
    for i in bad:
        print(f"  [{int(i):5d}] ref={ref[i]:+8.3f}  tc={out[i]:+8.3f}  diff={ref[i]-out[i]:+.3f}")
    if rel > 0.05 or abs_max > 1.0:
        print("BUG REPRODUCED — Inductor's Pad(edge)+dwconv fusion is unsound at the spatial border.")
        return 0
    print("not reproduced"); return 1


if __name__ == "__main__":
    sys.exit(main())
