#!/usr/bin/env python3
"""
RC-11h · Expand+Add+Mul with LayerNorm axis=1 fold

Single-case cluster (1 bug).  Representative = unique_0073.
Suspect backend(s): ['torch_compile', 'tvm'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(27)
SHAPE = (1, 4, 1, 1)


def build_model():
    B, C, H, W = SHAPE; D = C

    shape = np.array([1, C, 8, 8], np.int64)
    bias  = np.random.randn(1, C, 1, 1).astype(np.float32) * 0.01
    scl   = np.ones((1, C, 1, 1), np.float32)
    # LayerNorm axis=1 over X.shape = [1, C, 8, 8] -> normalized shape (C, 8, 8)
    gamma = np.ones((C, 8, 8), np.float32); beta = np.zeros((C, 8, 8), np.float32)
    nodes = [
    oh.make_node("Expand", ["x", "shape"], ["ex"]),
    oh.make_node("Add",    ["ex", "bias"], ["a1"]),
    oh.make_node("Mul",    ["a1", "scl"],  ["m1"]),
    oh.make_node("LayerNormalization", ["m1", "gamma", "beta"], ["y"], axis=1, epsilon=1e-5),
    ]
    inits = [onh.from_array(shape, "shape"), onh.from_array(bias, "bias"),
    onh.from_array(scl, "scl"), onh.from_array(gamma, "gamma"),
    onh.from_array(beta, "beta")]
    graph = oh.make_graph(
        nodes, "rc_11h_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['torch_compile', 'tvm']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
