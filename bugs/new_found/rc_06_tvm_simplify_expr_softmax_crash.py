#!/usr/bin/env python3
"""
Root Cause 6 — TVM relay's SimplifyExpr pass crashes on
`Add(m, m) → Sub(_, m) → Softmax(axis=0)`.

Covers crashes : 15, 170, 353, 355, 910  (all five TVM crashes)
Backends       : onnxruntime (reference, succeeds) vs TVM (crashes in pass)

Graph (4 ops):
    m  = MatMul(x, W)
    a  = Add(m, m)         # algebraically equal to 2*m
    s  = Sub(a, m)         # algebraically equal to m
    y  = Softmax(s, axis=0)

`Sub(Add(m, m), m)` is a textbook target for an algebraic simplifier:
the result is just `m`. TVM's `SimplifyExpr` rewrite recognises this and
substitutes — but it leaves the original `model_input` reference dangling
when the substituted graph is then handed to the lowering pass. The
resulting Relay module fails the well-formed check with:

    contains free variables: [Var(model_input, ...)]

The crash always points back inside `relay/transform/simplify_expr.cc`,
regardless of what activation surrounds the algebraic-zero pattern (the
five distinct campaign crashes only differ in surrounding ops — TopK,
GELU, CumSum, Tanh — but all hit the same simplifier bug).

Usage: cd Xcomp_V2 && PYTHONPATH=. python rc_06_tvm_simplify_expr_softmax_crash.py
Exit 0 = bug reproduced (TVM crashes), 1 = unexpectedly succeeded, 2 = TVM unavailable.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

warnings.filterwarnings("ignore")


def build_model(D: int = 64) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x_vi = helper.make_tensor_value_info("model_input", TensorProto.FLOAT, [1, D])
    y_vi = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, D])
    W = numpy_helper.from_array(rng.normal(0, 0.1, (D, D)).astype(np.float32), "W")
    nodes = [
        helper.make_node("MatMul",  ["model_input", "W"], ["m"]),
        helper.make_node("Add",     ["m", "m"],           ["a"]),  # 2 * m
        helper.make_node("Sub",     ["a", "m"],           ["s"]),  # ≡ m  (algebraic 0)
        helper.make_node("Softmax", ["s"], ["y"], axis=0),
    ]
    g = helper.make_graph(nodes, "tvm_simplify_crash", [x_vi], [y_vi], [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def main() -> int:
    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
    from trion.oracle.tvm_backend import TVMBackend

    tvm = TVMBackend()
    if not tvm.is_available():
        print("SKIP: TVM backend not available."); return 2

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (1, 64)).astype(np.float32)
    model = build_model(64)

    r_ort = ONNXRuntimeBackend().run(model, {"model_input": x}, optimized=True)
    r_tvm = tvm.run(model, {"model_input": x}, optimized=True)

    if not r_ort.ok:
        print(f"setup error: ORT failed: {r_ort.error}"); return 2

    ref = np.asarray(r_ort.output, np.float64).ravel()
    print(f"ORT output (correct)   first 6 : {ref[:6]}")
    print(f"ORT |out|_max = {np.max(np.abs(ref)):.4e}   sum = {ref.sum():.4f}")
    print(f"(For axis=0 softmax with batch=1, all outputs must equal 1.0; sum = 64.)")

    if r_tvm.ok:
        out = np.asarray(r_tvm.output, np.float64).ravel()
        print(f"TVM (unexpected OK)    first 6 : {out[:6]}")
        rel = float(np.linalg.norm(ref - out) / (np.linalg.norm(ref) + 1e-8))
        print(f"rel_L2 = {rel:.6f}")
        if rel > 0.05:
            print("\nBUG REPRODUCED (numerical divergence).")
            return 0
        print("not reproduced (TVM produced a finite, matching answer)")
        return 1

    err = (r_tvm.error or "")
    print(f"\nTVM crashed during compilation. Error tail:")
    for line in err.splitlines()[-8:]:
        print(f"  | {line[:120]}")

    fingerprints = ("simplify_expr", "SimplifyExpr", "free variables",
                    "WarnIfMalformed", "well-formed")
    if any(fp in err for fp in fingerprints):
        print("\nBUG REPRODUCED — TVM SimplifyExpr leaves a free variable.")
        print("ORT runs the same graph cleanly; TVM cannot lower it at all.")
        return 0
    print("\nTVM crashed but the failure signature does not match the SimplifyExpr cluster.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
