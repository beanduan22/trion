#!/usr/bin/env python3
"""
RC-11c · TF + Inductor Resize(linear, half_pixel) 5->4

Single-case cluster (1 bug).  Representative = unique_0014.
Suspect backend(s): ['tensorflow', 'torch_compile'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(22)
SHAPE = (1, 3, 5, 5)


def build_model():
    B, C, H, W = SHAPE; D = C

    sizes = np.array([1, C, 4, 4], np.int64)
    scales_empty = np.array([], np.float32); roi_empty = np.array([], np.float32)
    nodes = [
    oh.make_node("Relu",  ["x"], ["r"]),
    oh.make_node("Resize",
    ["r", "roi_empty", "scales_empty", "sizes"], ["y"],
    mode="linear", coordinate_transformation_mode="half_pixel"),
    ]
    inits = [onh.from_array(sizes, "sizes"),
    onh.from_array(scales_empty, "scales_empty"),
    onh.from_array(roi_empty,    "roi_empty")]
    graph = oh.make_graph(
        nodes, "rc_11c_min",
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
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['tensorflow', 'torch_compile']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
