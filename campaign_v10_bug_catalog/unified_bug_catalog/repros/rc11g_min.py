#!/usr/bin/env python3
"""
RC-11g · Spatial-attention (CBAM) reduce-max ambiguity

Single-case cluster (1 bug).  Representative = unique_0065.
Suspect backend(s): ['tensorflow', 'torchscript', 'xla'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(26)
SHAPE = (1, 4, 8, 8)


def build_model():
    B, C, H, W = SHAPE; D = C

    ax_c = np.array([1], np.int64)
    w    = np.random.randn(1, 2, 3, 3).astype(np.float32) * 0.1
    b    = np.zeros(1, np.float32)
    nodes = [
    oh.make_node("ReduceMean", ["x", "ax_c"], ["avg"], keepdims=1),
    oh.make_node("ReduceMax",  ["x", "ax_c"], ["mx"],  keepdims=1),
    oh.make_node("Concat",     ["avg", "mx"], ["cat"], axis=1),
    oh.make_node("Conv",       ["cat", "w", "b"], ["c1"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("Sigmoid",    ["c1"], ["s"]),
    oh.make_node("Mul",        ["s", "x"], ["y"]),
    ]
    inits = [onh.from_array(ax_c, "ax_c"), onh.from_array(w, "w"), onh.from_array(b, "b")]
    graph = oh.make_graph(
        nodes, "rc_11g_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['tensorflow', 'torchscript', 'xla']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
