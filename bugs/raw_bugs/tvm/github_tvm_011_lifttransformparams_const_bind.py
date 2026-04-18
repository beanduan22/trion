#!/usr/bin/env python3
# Run with: /home/binduan/miniconda3/envs/clawwork/bin/python bugs/new_minimal/tvm/github_tvm_011_lifttransformparams_const_bind.py
"""
Bug ID     : github_tvm_011
Source     : GitHub — apache/tvm Relay FoldConstant / algebraic-simplify
Compiler   : TVM Relay 0.11.1  vs  ONNX Runtime 1.23

Root cause
----------
The graph is ``Y = Mul(X, INF) - Mul(X, INF)`` where ``INF`` is a
constant initializer with every entry ``+inf``.  IEEE-754 semantics
(and what ORT implements) say the product is ``+inf`` for positive X
and ``-inf`` for negative X — subtracting two non-finite values of the
same sign yields ``NaN``, not zero.

TVM Relay's ``FoldConstant`` / algebraic-simplification passes see the
pattern ``a - a`` and rewrite it to a zero-filled constant with the
same shape before the constant ``INF`` is ever materialised, binding
the wrong value into the compiled graph.  ORT preserves IEEE arithmetic
and returns an all-NaN output.

This is the same class of bug described in apache/tvm#17207
(LiftTransformParams / const-binding) — the Relax flavour of the
constant-rebind problem.  The Relay 0.11 pipeline exhibits the
equivalent failure on the Relay side, reachable purely through the
ONNX frontend.

Exit codes
----------
0 = BUG REPRODUCED (TVM differs from ORT by > TOL)
1 = TVM matched ORT
2 = missing TVM or onnxruntime import
"""
from __future__ import annotations

import sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
except ImportError as e:
    print(f"missing dependency: {e}")
    sys.exit(2)


TOL: float = 1e-4


def build_model() -> onnx.ModelProto:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    inf_t = helper.make_tensor(
        "INF", TensorProto.FLOAT, [4], [float("inf")] * 4
    )
    nodes = [
        helper.make_node("Mul", ["X", "INF"], ["xa"]),
        helper.make_node("Sub", ["xa", "xa"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "g", [X], [Y], initializer=[inf_t])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )


def run_ort(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"X": x})[0]


def run_tvm(model: onnx.ModelProto, x: np.ndarray, opt_level: int = 3) -> np.ndarray:
    mod, params = relay.frontend.from_onnx(model, shape={"X": list(x.shape)})
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target="llvm", params=params)
    gm = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    gm.set_input("X", x)
    gm.run()
    return gm.get_output(0).numpy()


def main() -> int:
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    model = build_model()
    ort_out = run_ort(model, x)
    tvm_out = run_tvm(model, x, opt_level=3)

    print("input:", x)
    print("ORT  :", ort_out, "(expected all-NaN per IEEE-754)")
    print("TVM  :", tvm_out, "(expected to match — but folds a-a to 0)")

    # NaN-aware difference: treat NaN as mismatching anything.
    nan_mask = np.isnan(ort_out) ^ np.isnan(tvm_out)
    finite_diff = np.abs(
        np.nan_to_num(ort_out, nan=0.0).astype(np.float64)
        - np.nan_to_num(tvm_out, nan=0.0).astype(np.float64)
    )
    # Force a large "distance" on any NaN mismatch.
    finite_diff[nan_mask] = 1.0
    max_abs = float(finite_diff.max())
    print(f"max_abs={max_abs:.6e}  tol={TOL:.0e}  (NaN mismatch counted as 1.0)")

    if max_abs > TOL:
        print(
            "BUG REPRODUCED: TVM Relay FoldConstant rewrites "
            "inf*X - inf*X to 0, violating IEEE-754 (ORT returns NaN)"
        )
        return 0
    print("not reproduced (TVM matched ORT)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
