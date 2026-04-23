#!/usr/bin/env python3
"""
RC-11a · Row-reduce + Mul + Transpose + Softmax + LayerNorm misfold

Single-case cluster (1 bug).  Representative = unique_0004.
Suspect backend(s): ['openvino', 'torchscript', 'tvm', 'xla'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(20)
SHAPE = (1, 8, 8)


def build_model():
    B, N, D = SHAPE; C = N

    axes_last = np.array([-1], np.int64)
    gamma = np.ones(D, np.float32); beta = np.zeros(D, np.float32)
    W1 = np.random.randn(D, D).astype(np.float32)
    nodes = [
    oh.make_node("ReduceMean", ["x", "axes_last"], ["m"], keepdims=1),
    oh.make_node("Mul",        ["x", "m"],         ["mx"]),
    oh.make_node("Transpose",  ["mx"], ["tx"], perm=[0, 2, 1]),
    oh.make_node("MatMul",     ["tx", "W1"], ["mm"]),
    oh.make_node("Softmax",    ["mm"], ["sm"], axis=-1),
    oh.make_node("LayerNormalization", ["sm", "gamma", "beta"], ["y"], axis=-1, epsilon=1e-5),
    ]
    inits = [onh.from_array(axes_last, "axes_last"), onh.from_array(gamma, "gamma"),
    onh.from_array(beta, "beta"), onh.from_array(W1, "W1")]
    graph = oh.make_graph(
        nodes, "rc_11a_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, list(SHAPE))],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=inits,
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(*SHAPE).astype(np.float32)}
    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref:
        print(f"[setup] ORT failed: {err_ref}"); return 2
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['openvino', 'torchscript', 'tvm', 'xla']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
