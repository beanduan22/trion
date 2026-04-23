#!/usr/bin/env python3
"""
CR-01 · TVM Relay build crashes with "contains free variables" when an
intermediate tensor is used as BOTH inputs of a binary op and the model is
built with ``freeze_params=True``.

Cluster: 5 unique reproducible TVM backend crashes from campaign_v10 (models
385, 905, 959, 1225, 1452). In all five cases the ONNX graph references
``model_input`` exactly once, but TVM's Relay build pipeline ends up with
multiple ``Var(model_input, …)`` in the final function body — the free-var
check aborts.

Minimal repro: **2 ONNX nodes** — ``MatMul(x, W) → Add(m, m)``. With
``freeze_params=True``, TVM 0.11.1 aborts during ``relay.build`` with
"contains free variables".

Run:
    python crash_root_cause_catalog/repros/cr01_tvm_free_vars_min.py

Exit 0 = crash reproduced, 1 = build succeeded (bug gone), 2 = setup error.
"""
from __future__ import annotations

import sys
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh


def build_min_model():
    """Build the minimal 2-node ONNX graph that triggers the crash."""
    D = 4
    W = np.random.randn(D, D).astype(np.float32)
    nodes = [
        oh.make_node("MatMul", ["x", "W"], ["m"]),
        # Self-add: the bug trigger. `m` is used as both inputs.
        oh.make_node("Add",    ["m", "m"], ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "cr01_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [1, D])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[onh.from_array(W, "W")],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])


def main() -> int:
    try:
        import tvm
        from tvm import relay
    except ImportError:
        print("[setup] TVM is not importable in this interpreter; "
              "use e.g. miniconda3/envs/clawwork which ships tvm 0.11.1.")
        return 2

    np.random.seed(0)
    model = build_min_model()
    onnx.checker.check_model(model)

    shape = {"x": [1, 4]}
    mod, params = relay.frontend.from_onnx(model, shape, freeze_params=True)

    try:
        with tvm.transform.PassContext(opt_level=0):
            relay.build(mod, target="llvm", params=params)
    except Exception as e:
        txt = str(e)
        if "free variables" in txt or "fv.size()" in txt:
            tail = txt.rsplit("\n", 2)[-1][:200]
            print(f"[REPRO] CR-01 triggered — TVM aborted with: {tail}")
            return 0
        print(f"[unexpected] different TVM error: {txt[-200:]}")
        return 2

    print("[not-repro] relay.build succeeded — the bug may be fixed in this TVM version.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
