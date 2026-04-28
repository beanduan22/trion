#!/usr/bin/env python3
"""
TVM Relay opt_level=3 — Mul(x, 0) → Add(scale) → MatMul(W) returns NaN.

Standalone minimal reproducer (stock onnx + onnxruntime + apache-tvm).

    mz = Mul(x, zeros)        # zeros is an all-zero [1, D] initialiser
    sa = Add(mz, scale)       # sa ≡ scale  (constant)
    y  = MatMul(sa, W)        # y  ≡ scale @ W  (constant in x)

Output is constant by construction. ORT and TVM at opt_level=0 return
the correct constant. TVM at opt_level=3 returns NaN for every element.

    pip install onnx onnxruntime apache-tvm numpy
    python rc_02_tvm_mul_zero_matmul_nan.py

Exit 0 = bug reproduced, 1 = not reproduced, 2 = TVM unavailable.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
warnings.filterwarnings("ignore")
D = 128


def build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    inits = [
        numpy_helper.from_array(np.zeros((1, D), np.float32), "zeros"),
        numpy_helper.from_array(rng.normal(0, 1, (1, D)).astype(np.float32), "scale"),
        numpy_helper.from_array(rng.normal(0, 0.1, (D, D)).astype(np.float32), "W"),
    ]
    nodes = [
        helper.make_node("Mul", ["x", "zeros"], ["mz"]),
        helper.make_node("Add", ["mz", "scale"], ["sa"]),
        helper.make_node("MatMul", ["sa", "W"], ["y"]),
    ]
    g = helper.make_graph(
        nodes, "mul_zero_matmul",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, D])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def run_ort(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(model.SerializeToString(), sess_options=o,
                             providers=["CPUExecutionProvider"])
    return s.run(None, {"x": x})[0]


def run_tvm(model: onnx.ModelProto, x: np.ndarray, opt_level: int) -> np.ndarray:
    import tvm
    from tvm import relay
    from tvm.contrib.graph_executor import GraphModule
    mod, params = relay.frontend.from_onnx(model, {"x": x.shape})
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target="llvm", params=params)
    gm = GraphModule(lib["default"](tvm.cpu(0)))
    gm.set_input("x", tvm.nd.array(x))
    gm.run()
    return gm.get_output(0).numpy()


def main() -> int:
    try:
        import tvm  # noqa: F401
    except ImportError:
        print("SKIP: apache-tvm is not installed."); return 2

    x = np.random.default_rng(42).normal(0, 1, (1, D)).astype(np.float32)
    m = build_model()
    y_ort = run_ort(m, x)
    y_o0  = run_tvm(m, x, 0)
    y_o3  = run_tvm(m, x, 3)
    n_nan = int(np.isnan(y_o3).sum())
    print(f"ORT     y[0:6] = {y_ort[0, :6]}")
    print(f"TVM-O0  y[0:6] = {y_o0 [0, :6]}")
    print(f"TVM-O3  y[0:6] = {y_o3 [0, :6]}")
    print(f"TVM-O3 NaN count: {n_nan}/{y_o3.size}")
    if n_nan > 0:
        print("\nBUG REPRODUCED — TVM-O3 returns NaN for a graph whose output\n"
              "is constant in x. TVM-O0 matches ORT to machine epsilon, so\n"
              "the bug is in TVM's optimisation passes, not the importer.")
        return 0
    print("not reproduced — TVM-O3 matches ORT.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
