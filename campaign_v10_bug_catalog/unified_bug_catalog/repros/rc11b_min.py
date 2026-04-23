#!/usr/bin/env python3
"""
RC-11b · Reciprocal(zero) + Conv + BN fuse corner

Single-case cluster (1 bug).  Representative = unique_0007.
Suspect backend(s): ['openvino', 'torchscript', 'tvm'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(21)
SHAPE = (1, 4, 8, 8)


def build_model():
    B, C, H, W = SHAPE; D = C

    w = np.random.randn(C, C, 3, 3).astype(np.float32) * 0.1
    b = np.zeros(C, np.float32)
    bn_s = np.ones(C, np.float32); bn_b = np.zeros(C, np.float32)
    bn_m = np.zeros(C, np.float32); bn_v = np.ones(C, np.float32)
    nodes = [
    oh.make_node("Reciprocal", ["x"],       ["r"]),
    oh.make_node("Mul",        ["r", "x"],  ["rx"]),
    oh.make_node("Conv",       ["rx", "w", "b"], ["c"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("BatchNormalization",
    ["c", "bn_s", "bn_b", "bn_m", "bn_v"], ["bn"], epsilon=1e-5),
    oh.make_node("Relu",       ["bn"], ["y"]),
    ]
    inits = [onh.from_array(w, "w"), onh.from_array(b, "b"),
    onh.from_array(bn_s, "bn_s"), onh.from_array(bn_b, "bn_b"),
    onh.from_array(bn_m, "bn_m"), onh.from_array(bn_v, "bn_v")]
    graph = oh.make_graph(
        nodes, "rc_11b_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['openvino', 'torchscript', 'tvm']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
