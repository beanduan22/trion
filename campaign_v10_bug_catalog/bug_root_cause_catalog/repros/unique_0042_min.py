#!/usr/bin/env python3
"""
RC-09 · XLA-only Resize + 4-D batched-MatMul layout lowering.

Cluster: 3 unique bugs.  rel_l2 = 1.000 on all three.
Representative source: unique_0042 (campaign bug #1182).

Trigger pattern:
    x -> MatMul(4D) -> Slice(step=2) -> Tile -> MatMul -> Mul -> Add -> Relu
      -> Resize(nearest, asymmetric)

Reference cluster: every backend including tensorflow.
Suspect          : xla only.

TF (which usually goes through XLA) matches the reference — this is the
standalone XLA backend, not the TF-XLA bridge.  Standalone XLA lays out the
4-D batched matmul result differently before Resize, so the final image is
totally off (hence rel_l2 = 1.0).
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, C, H, W = 1, 2, 8, 8
TOL = 1e-3
np.random.seed(9)


def build_model():
    W1 = np.random.randn(W, W).astype(np.float32) * 0.1
    W2 = np.random.randn(W, W).astype(np.float32) * 0.1
    s_starts = np.array([0], dtype=np.int64)
    s_ends   = np.array([H], dtype=np.int64)
    s_axes   = np.array([2], dtype=np.int64)
    s_steps  = np.array([2], dtype=np.int64)
    tile_rep = np.array([1, 1, 2, 1], dtype=np.int64)
    scale = np.array([[[[0.5]]]], dtype=np.float32)
    bias  = np.array([[[[0.1]]]], dtype=np.float32)
    sizes = np.array([B, C, 2 * H, 2 * W], dtype=np.int64)
    scales_empty = np.array([], dtype=np.float32)
    roi_empty    = np.array([], dtype=np.float32)

    nodes = [
        oh.make_node("MatMul", ["x", "W1"], ["mm1"]),
        oh.make_node("Slice",  ["mm1", "s_starts", "s_ends", "s_axes", "s_steps"], ["sl"]),
        oh.make_node("Tile",   ["sl", "tile_rep"], ["tl"]),
        oh.make_node("MatMul", ["tl", "W2"], ["mm2"]),
        oh.make_node("Mul",    ["mm2", "scale"], ["ms"]),
        oh.make_node("Add",    ["ms", "bias"],  ["ad"]),
        oh.make_node("Relu",   ["ad"], ["rl"]),
        oh.make_node("Resize",
                     ["rl", "roi_empty", "scales_empty", "sizes"], ["y"],
                     mode="nearest", coordinate_transformation_mode="asymmetric"),
    ]
    graph = oh.make_graph(
        nodes, "rc09_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, C, H, W])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[onh.from_array(W1, "W1"), onh.from_array(W2, "W2"),
                     onh.from_array(s_starts, "s_starts"), onh.from_array(s_ends, "s_ends"),
                     onh.from_array(s_axes, "s_axes"), onh.from_array(s_steps, "s_steps"),
                     onh.from_array(tile_rep, "tile_rep"),
                     onh.from_array(scale, "scale"), onh.from_array(bias, "bias"),
                     onh.from_array(sizes, "sizes"),
                     onh.from_array(scales_empty, "scales_empty"),
                     onh.from_array(roi_empty,    "roi_empty")],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, C, H, W).astype(np.float32)}
    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2
    xla, _ = run_backend("xla", model, inputs)
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=[("xla", xla)],
                     outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
