#!/usr/bin/env python3
"""
RC-11k · TorchScript-only ReduceL2 + manual LayerNorm

Single-case cluster (1 bug).  Representative = unique_0108.
Suspect backend(s): ['torchscript'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(30)
SHAPE = (1, 8, 8)


def build_model():
    B, N, D = SHAPE; C = N

    ax_last = np.array([-1], np.int64)
    eps_v  = np.array([1e-5], np.float32)
    nodes = [
    oh.make_node("Mul",       ["x", "x"], ["sq"]),
    oh.make_node("ReduceSum", ["sq", "ax_last"], ["ssum"], keepdims=1),
    oh.make_node("Add",       ["ssum", "eps_v"], ["se"]),
    oh.make_node("Sqrt",      ["se"], ["rm"]),
    oh.make_node("Div",       ["x", "rm"], ["y"]),
    ]
    inits = [onh.from_array(ax_last, "ax_last"), onh.from_array(eps_v, "eps_v")]
    graph = oh.make_graph(
        nodes, "rc_11k_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['torchscript']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
