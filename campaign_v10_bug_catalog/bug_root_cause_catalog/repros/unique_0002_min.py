#!/usr/bin/env python3
"""
RC-10 · Resize nearest-asymmetric with `round_prefer_floor`.

Cluster: 2 unique bugs (cluster signature is the 2-bug set, but the
underlying rounding-rule issue is shared by a further 3 RC-08 bugs that add
depthwise-conv on top).  rel_l2 = 1.000 on both bugs in this cluster.
Representative source: unique_0002 (campaign bug #287).

Trigger pattern:
    x -> BatchNormalization -> Mul -> Div -> Add
      -> Resize(mode=nearest, coordinate_transformation=asymmetric,
                nearest_mode=round_prefer_floor, scale = 26/64)
      -> Mul -> Greater -> Where -> Tanh -> Add -> Mul

Reference cluster: pytorch_eager + onnxruntime + openvino.
Suspect          : tensorflow, torch_compile, torchscript, tvm, xla.

The ONNX spec states that nearest mode has four variants, and
`round_prefer_floor` is the opset-13+ default.  OV + ORT implement it
literally; the other five backends collapse to either 'always-floor' or
'round-half-to-even'.  With a non-integer output ratio (e.g. 64→26), the
chosen pixel column differs by 1 for ~half the indices → rel_l2 = 1.0.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, C, H, W = 1, 4, 26, 26
OH, OW = 64, 64
TOL = 1e-3
np.random.seed(10)


def build_model():
    bn_s = np.ones(C, dtype=np.float32) * 0.9
    bn_b = np.zeros(C, dtype=np.float32)
    bn_m = np.zeros(C, dtype=np.float32)
    bn_v = np.ones(C, dtype=np.float32)
    scale = np.array([[[[0.5]]]], dtype=np.float32)
    div   = np.array([[[[1.3]]]], dtype=np.float32)
    bias  = np.array([[[[0.1]]]], dtype=np.float32)
    sizes = np.array([B, C, OH, OW], dtype=np.int64)
    scales_empty = np.array([], dtype=np.float32)
    roi_empty    = np.array([], dtype=np.float32)
    thr = np.array([0.0], dtype=np.float32)
    fill = np.array([1.0], dtype=np.float32)

    nodes = [
        oh.make_node("BatchNormalization",
                     ["x", "bn_s", "bn_b", "bn_m", "bn_v"], ["bn"], epsilon=1e-5),
        oh.make_node("Mul", ["bn", "scale"], ["m1"]),
        oh.make_node("Div", ["m1", "div"],   ["d1"]),
        oh.make_node("Add", ["d1", "bias"],  ["a1"]),
        oh.make_node("Resize",
                     ["a1", "roi_empty", "scales_empty", "sizes"], ["rz"],
                     mode="nearest",
                     coordinate_transformation_mode="asymmetric",
                     nearest_mode="round_prefer_floor"),
        oh.make_node("Mul",     ["rz", "scale"], ["rm"]),
        oh.make_node("Greater", ["rm", "thr"],   ["g"]),
        oh.make_node("Where",   ["g", "rm", "fill"], ["w"]),
        oh.make_node("Tanh",    ["w"], ["t"]),
        oh.make_node("Add",     ["t", "bias"], ["a2"]),
        oh.make_node("Mul",     ["a2", "scale"], ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "rc10_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, C, H, W])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[onh.from_array(bn_s, "bn_s"), onh.from_array(bn_b, "bn_b"),
                     onh.from_array(bn_m, "bn_m"), onh.from_array(bn_v, "bn_v"),
                     onh.from_array(scale, "scale"), onh.from_array(div, "div"),
                     onh.from_array(bias, "bias"), onh.from_array(sizes, "sizes"),
                     onh.from_array(scales_empty, "scales_empty"),
                     onh.from_array(roi_empty,    "roi_empty"),
                     onh.from_array(thr, "thr"), onh.from_array(fill, "fill")],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, C, H, W).astype(np.float32)}
    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2
    outs = [(name, run_backend(name, model, inputs)[0])
            for name in ["tensorflow", "torch_compile",
                         "torchscript", "tvm", "xla"]]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
