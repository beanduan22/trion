#!/usr/bin/env python3
"""
RC-11e · foldable add-zero/mul-one erases Where-const branch

Single-case cluster (1 bug).  Representative = unique_0050.
Suspect backend(s): ['tensorflow', 'torch_compile', 'torchscript'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(24)
SHAPE = (1, 8, 8)


def build_model():
    B, N, D = SHAPE; C = N

    zero = np.array([[0.0]*D], np.float32)
    one  = np.array([[1.0]*D], np.float32)
    thr  = np.array([0.0], np.float32)
    fill = np.array([[1.0]*D], np.float32)
    W = np.random.randn(D, D).astype(np.float32)
    nodes = [
    oh.make_node("MatMul",  ["x", "W"], ["m1"]),
    oh.make_node("MatMul",  ["m1", "W"], ["m2"]),
    oh.make_node("MatMul",  ["m2", "W"], ["m3"]),
    oh.make_node("Add",     ["m3", "zero"], ["a"]),
    oh.make_node("Mul",     ["a", "one"],   ["mu"]),
    oh.make_node("Greater", ["mu", "thr"], ["g"]),
    oh.make_node("Where",   ["g", "mu", "fill"], ["y"]),
    ]
    inits = [onh.from_array(zero, "zero"), onh.from_array(one, "one"),
    onh.from_array(thr, "thr"), onh.from_array(fill, "fill"),
    onh.from_array(W, "W")]
    graph = oh.make_graph(
        nodes, "rc_11e_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['tensorflow', 'torch_compile', 'torchscript']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
