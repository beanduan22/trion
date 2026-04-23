#!/usr/bin/env python3
"""
RC-11d · branch Min/Max + LayerNorm residual

Single-case cluster (1 bug).  Representative = unique_0035.
Suspect backend(s): ['torchscript', 'tvm', 'xla'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(23)
SHAPE = (1, 4, 16)


def build_model():
    B, N, D = SHAPE; C = N

    c1 = np.array([-2.0], np.float32); c2 = np.array([2.0], np.float32)
    gamma = np.ones(D, np.float32); beta = np.zeros(D, np.float32)
    nodes = [
    oh.make_node("Max", ["x", "c1"], ["hi"]),
    oh.make_node("Min", ["hi", "c2"], ["cl"]),
    oh.make_node("LayerNormalization", ["cl", "gamma", "beta"], ["ln"], axis=-1, epsilon=1e-5),
    oh.make_node("Add", ["ln", "x"], ["res"]),
    oh.make_node("Relu",["res"], ["y"]),
    ]
    inits = [onh.from_array(c1, "c1"), onh.from_array(c2, "c2"),
    onh.from_array(gamma, "gamma"), onh.from_array(beta, "beta")]
    graph = oh.make_graph(
        nodes, "rc_11d_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['torchscript', 'tvm', 'xla']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
