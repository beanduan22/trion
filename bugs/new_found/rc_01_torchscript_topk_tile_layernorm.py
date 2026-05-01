#!/usr/bin/env python3
"""
Root Cause 1 — TorchScript and TVM/ORT diverge on LayerNorm of a constant
row produced by `TopK(k=1) → Tile`.

Covers bugs : 75, 589, 920  (all three "TS rel_L2 ≈ 1" bugs)
Backends    : onnxruntime (reference) vs torchscript (onnx2torch + torch.jit.trace)

Graph (3 nodes):
    vals, _ = TopK(x, k=1, axis=-1, largest=1, sorted=1)   # shape [N, 1]
    tiled   = Tile(vals, repeats=[1, D])                   # shape [N, D]
    y       = LayerNormalization(tiled, axis=-1, eps=1e-5)

Mathematically every row of `tiled` is constant (= max of the row), so
LayerNorm should output exactly zero. In fp32 the answer depends on
which kernel computes the variance:
  * ORT's `LayerNormalization` op fuses `(x - mean)/sqrt(var+eps)` and
    rounds to a tiny non-zero residue (~1e-3 for D=128/256).
  * onnx2torch lowers to `F.layer_norm`, which computes variance via
    Welford and rounds the residue to *exactly* zero.

Both are mathematically valid. The campaign's relative-L2 oracle
divides by `||ORT||` and reports 1.0 because TS produces 0 — yet the
absolute difference is ~5e-4. So the divergence flagged by the fuzzer
is real (kernel-level disagreement on a degenerate input) but the
output magnitudes are at the noise floor — not a wildly-wrong-output
bug, but a precision-stability bug at the LayerNorm kernel boundary
that surfaces whenever a TopK(k=1)+Tile produces a constant row.

Usage: cd Xcomp_V2 && PYTHONPATH=. python rc_01_torchscript_topk_tile_layernorm.py
Exit 0 = bug reproduced, 1 = not reproduced.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

warnings.filterwarnings("ignore")


def build_model(N: int = 1, D: int = 256) -> onnx.ModelProto:
    x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, D])
    y_vi = helper.make_tensor_value_info("y", TensorProto.FLOAT, [N, D])
    inits = [
        numpy_helper.from_array(np.array([1], np.int64), "k"),
        numpy_helper.from_array(np.array([1, D], np.int64), "rep"),
        numpy_helper.from_array(np.ones (D, np.float32), "ln_scale"),
        numpy_helper.from_array(np.zeros(D, np.float32), "ln_bias"),
    ]
    nodes = [
        helper.make_node("TopK", ["x", "k"], ["vals", "idx"],
                         axis=-1, largest=1, sorted=1),
        helper.make_node("Tile", ["vals", "rep"], ["tiled"]),
        helper.make_node("LayerNormalization",
                         ["tiled", "ln_scale", "ln_bias"], ["y"],
                         axis=-1, epsilon=1e-5),
    ]
    g = helper.make_graph(nodes, "topk_tile_ln", [x_vi], [y_vi], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def main() -> int:
    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
    from trion.oracle.torchscript_backend import TorchScriptBackend

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1, 256)).astype(np.float32)
    model = build_model(1, 256)

    r_ort = ONNXRuntimeBackend().run(model, {"x": x}, optimized=True)
    r_ts  = TorchScriptBackend().run(model, {"x": x}, optimized=True)
    if not r_ort.ok:
        print(f"setup error: ORT failed: {r_ort.error}"); return 2
    ref = np.asarray(r_ort.output, np.float64).ravel()
    if not r_ts.ok:
        print(f"TorchScript error: {r_ts.error}"); return 0
    out = np.asarray(r_ts.output, np.float64).ravel()

    print(f"input row max  = {x.max():.4f}    (LN is applied to that constant tiled across D=256)")
    print(f"ORT (first 5)  = {ref[:5]}   ||ref||_inf = {np.max(np.abs(ref)):.3e}")
    print(f"TS  (first 5)  = {out[:5]}   ||ts ||_inf = {np.max(np.abs(out)):.3e}")

    rel = float(np.linalg.norm(ref - out) / (np.linalg.norm(ref) + 1e-8))
    ad  = float(np.max(np.abs(ref - out)))
    print(f"rel_L2(TS, ORT)      = {rel:.6f}     <-- the fuzzer's metric")
    print(f"max_abs_diff(TS, ORT) = {ad:.3e}     <-- the absolute disagreement")
    print(f"TS produces exactly zero ({(out == 0).all()}); ORT keeps an fp32 residue.")

    if rel > 0.5 and ad < 1.0:
        print("\nBUG REPRODUCED — kernel-level fp32 precision divergence between")
        print("ORT's fused LayerNorm and torch's `F.layer_norm` on a degenerate")
        print("constant-row input. Output magnitudes are at the noise floor, but")
        print("the relative-L2 oracle reports 100% divergence. This is a real")
        print("backend disagreement; flag as severity-low (precision, not value).")
        return 0
    if rel > 0.5:
        print("BUG REPRODUCED — TS output deviates significantly from ORT.")
        return 0
    print("not reproduced")
    return 1


if __name__ == "__main__":
    sys.exit(main())
