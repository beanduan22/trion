#!/usr/bin/env python3
"""
Root Cause 2 — TVM produces NaN for Mul(x, 0) → Add(scale) → MatMul(W).

Covers bug  : 667  (and contributes to 585's catastrophic failure)
Backends    : onnxruntime (reference) vs TVM (relay.build opt_level=3)

Graph (3 nodes, no input-dependent values after the Mul-by-zero):
    mz = Mul(x, zeros)            # zeros is an all-zero [1, D] initializer
    sa = Add(mz, scale)           # sa ≡ scale, a constant
    y  = MatMul(sa, W)            # y ≡ scale @ W, a constant

Because `mz = 0` everywhere, the final output must equal `scale @ W` for
*any* value of `x`. ORT computes this correctly. TVM produces NaN.

Root cause: TVM's algebraic simplifier appears to rewrite `Mul(_, 0)` to a
broadcast-of-zero that loses shape metadata; the downstream `Add` then
propagates an operand whose shape is incompatible with the MatMul, and the
generated kernel reads uninitialized scratch memory (hence the NaNs).

Usage: cd Xcomp_V2 && PYTHONPATH=. python rc_02_tvm_mul_zero_matmul_nan.py
Exit 0 = bug reproduced, 1 = not reproduced, 2 = TVM unavailable.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

warnings.filterwarnings("ignore")


def build_model(D: int = 128) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])
    y_vi = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, D])
    zeros = numpy_helper.from_array(np.zeros((1, D), np.float32), "zeros")
    scale = numpy_helper.from_array(rng.normal(0, 1, (1, D)).astype(np.float32), "scale")
    W     = numpy_helper.from_array(rng.normal(0, 0.1, (D, D)).astype(np.float32), "W")
    nodes = [
        helper.make_node("Mul",    ["x", "zeros"],  ["mz"]),
        helper.make_node("Add",    ["mz", "scale"], ["sa"]),
        helper.make_node("MatMul", ["sa", "W"],     ["y"]),
    ]
    g = helper.make_graph(nodes, "mul_zero_matmul", [x_vi], [y_vi],
                          [zeros, scale, W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def main() -> int:
    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
    from trion.oracle.tvm_backend import TVMBackend

    tvm = TVMBackend()
    if not tvm.is_available():
        print("SKIP: TVM backend not available on this machine.")
        return 2

    rng = np.random.default_rng(42)
    D = 128
    x = rng.normal(0, 1, (1, D)).astype(np.float32)

    model = build_model(D)
    r_ort = ONNXRuntimeBackend().run(model, {"x": x}, optimized=True)
    r_tvm = tvm.run(model, {"x": x}, optimized=True)

    if not r_ort.ok:
        print(f"setup error: ORT failed: {r_ort.error}"); return 2

    ref = np.asarray(r_ort.output, np.float64).ravel()
    print(f"ORT output (correct)   first 6: {ref[:6]}")
    print(f"ORT |out|_max = {np.max(np.abs(ref)):.4e}   any NaN? {np.isnan(ref).any()}")

    if not r_tvm.ok:
        print(f"TVM backend raised   : {r_tvm.error}")
        print("→ TVM crashed where ORT produced a well-defined finite output.")
        return 0

    out = np.asarray(r_tvm.output, np.float64).ravel()
    n_nan = int(np.isnan(out).sum())
    print(f"TVM output (buggy)     first 6: {out[:6]}")
    print(f"TVM |out|_max = {np.nan_to_num(np.abs(out).max()):.4e}   NaN count = {n_nan}/{out.size}")

    if n_nan > 0:
        print("\nBUG REPRODUCED — TVM returns NaN for a graph whose output must")
        print("equal the constant `scale @ W` regardless of the input x.")
        print("ORT delivers finite values; TVM does not.")
        return 0

    rel = float(np.linalg.norm(ref - out) / (np.linalg.norm(ref) + 1e-8))
    print(f"rel_L2(TVM, ORT) = {rel:.6f}")
    if rel > 0.05:
        print("BUG REPRODUCED (numerical divergence, no NaN).")
        return 0
    print("not reproduced")
    return 1


if __name__ == "__main__":
    sys.exit(main())
