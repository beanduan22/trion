#!/usr/bin/env python3
"""
RC-11i · expm1(bounded) + attention + resize layout

Single-case cluster (1 bug).  Representative = unique_0076.
Suspect backend(s): ['torchscript', 'xla'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(28)
SHAPE = (1, 8, 8)


def build_model():
    B, N, D = SHAPE; C = N

    one = np.array([[1.0]*D], np.float32)
    W   = np.random.randn(D, D).astype(np.float32) * 0.1
    nodes = [
    oh.make_node("Exp",    ["x"], ["ex"]),
    oh.make_node("Sub",    ["ex", "one"], ["em1"]),
    oh.make_node("MatMul", ["em1", "W"], ["mm"]),
    oh.make_node("Tanh",   ["mm"], ["y"]),
    ]
    inits = [onh.from_array(one, "one"), onh.from_array(W, "W")]
    graph = oh.make_graph(
        nodes, "rc_11i_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, list(SHAPE))],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=inits,
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(*SHAPE).astype(np.float32)}
    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref:
        print(f"[setup] ORT failed: {err_ref}"); return 2
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['torchscript', 'xla']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
