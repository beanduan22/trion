#!/usr/bin/env python3
# Run with: /home/binduan/miniconda3/envs/clawwork/bin/python bugs/new_minimal/tvm/github_tvm_012_gelu_approx_tanh.py
"""
Bug ID     : github_tvm_012
Source     : GitHub — apache/tvm Relay ONNX importer, Gelu (opset 20)
Compiler   : TVM Relay 0.11.1  vs  ONNX Runtime 1.23

Root cause
----------
ONNX opset 20 promoted ``Gelu`` to a first-class op with an attribute
``approximate`` that takes the values ``"none"`` (exact, via ``erf``)
and ``"tanh"`` (the Hendrycks/Gimbel tanh approximation used in BERT,
GPT-2, LLaMA, etc.).  The tanh formula is

    gelu_tanh(x) = 0.5 * x *
                   (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

and differs from the exact ``erf`` formulation by up to ~4e-4 in the
range of interest for LLM activations (|x| ≤ 3).

TVM Relay's ONNX importer in 0.11.x routes *both* ``approximate="none"``
and ``approximate="tanh"`` to the same ``relay.nn.gelu`` implementation
(exact erf form) and silently drops the attribute.  For opset-20 models
whose intent is the tanh approximation, TVM therefore returns the exact
output — diverging from ORT by ~4e-4 everywhere, which exceeds the
1e-4 tolerance the TVM test suite uses for activation primitives.

For ``approximate="none"`` (exact) TVM and ORT agree — that case is the
control and should report ``max_abs≈0``.  The reported bug is the tanh
path.

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


def build_model(approximate: str) -> onnx.ModelProto:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [7])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [7])
    node = helper.make_node("Gelu", ["X"], ["Y"], approximate=approximate)
    graph = helper.make_graph([node], "g", [X], [Y])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
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
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    # Control: exact (approximate="none") — expected to match.
    model_exact = build_model("none")
    ort_exact = run_ort(model_exact, x)
    tvm_exact = run_tvm(model_exact, x, opt_level=3)
    ctl = float(np.max(np.abs(ort_exact - tvm_exact)))
    print("-- Control: approximate='none' (exact erf) --")
    print(f"  ORT: {ort_exact}")
    print(f"  TVM: {tvm_exact}")
    print(f"  max_abs={ctl:.6e}")

    # Target: tanh approximation — expected to diverge.
    model_tanh = build_model("tanh")
    ort_tanh = run_ort(model_tanh, x)
    tvm_tanh = run_tvm(model_tanh, x, opt_level=3)
    print("-- Target: approximate='tanh' --")
    print(f"  ORT: {ort_tanh}")
    print(f"  TVM: {tvm_tanh}")
    diff = np.abs(ort_tanh.astype(np.float64) - tvm_tanh.astype(np.float64))
    max_abs = float(diff.max())
    print(f"  max_abs={max_abs:.6e}  tol={TOL:.0e}")

    if max_abs > TOL:
        print(
            "BUG REPRODUCED: TVM Relay Gelu importer ignores "
            "approximate='tanh' and returns the exact erf output"
        )
        return 0
    print("not reproduced (TVM matched ORT for tanh approximation)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
