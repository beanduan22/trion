#!/usr/bin/env python3
"""
RC-11j · Reciprocal(-0.0) sign in OV + XLA

Single-case cluster (1 bug).  Representative = unique_0107.
Suspect backend(s): ['openvino', 'xla'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(29)
SHAPE = (1, 4)


def build_model():
    B, D = SHAPE; C = D; N = D

    zero_neg = np.array([[-0.0] * D], np.float32)  # concatenated to input
    nodes = [
    oh.make_node("Reciprocal", ["x"], ["r"]),
    oh.make_node("Relu",       ["r"], ["y"]),
    ]
    inits = []
    graph = oh.make_graph(
        nodes, "rc_11j_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['openvino', 'xla']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
