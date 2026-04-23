#!/usr/bin/env python3
"""
RC-07 · TopK(k=1) + Tile + downstream LayerNorm axis/rank mismatch.

Cluster: 4 unique bugs.  Every bug in the cluster hits rel_l2 = 1.000.
Representative source: unique_0074 (campaign bug #312).

Trigger pattern:
    x -> TopK(k=1, axis=-1) -> Tile -> Add -> Mul -> Concat
      -> Exp -> Slice -> Mul -> LayerNormalization -> Add -> Relu -> Sub

Reference cluster: pytorch_eager + onnxruntime + tvm.
Suspect          : openvino, tensorflow, torch_compile, torchscript, xla.

TopK with k=1 yields a singleton last-dim.  Five backends squeeze-or-keep
that dim inconsistently before LayerNorm, so LayerNorm picks a different
normalization axis than the spec.  rel_l2 is always 1.0 because the
normalization is applied along the wrong axis.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, D = 1, 16
TOL = 1e-3
np.random.seed(7)


def build_model():
    K = np.array([1], dtype=np.int64)
    tile_rep = np.array([1, D], dtype=np.int64)  # bring last dim back up
    s_starts = np.array([0], dtype=np.int64)
    s_ends   = np.array([D], dtype=np.int64)
    s_axes   = np.array([-1], dtype=np.int64)
    c1 = np.random.randn(1, D).astype(np.float32) * 0.1
    c2 = np.random.randn(1, D).astype(np.float32) * 0.1
    c3 = np.random.randn(1, D).astype(np.float32) * 0.05
    gamma = np.ones(D, dtype=np.float32)
    beta  = np.zeros(D, dtype=np.float32)

    nodes = [
        oh.make_node("TopK",  ["x", "K"], ["vals", "inds"], axis=-1, largest=1, sorted=1),
        oh.make_node("Tile",  ["vals", "tile_rep"], ["tl"]),
        oh.make_node("Add",   ["tl", "c1"], ["a1"]),
        oh.make_node("Mul",   ["a1", "c2"], ["m1"]),
        oh.make_node("Concat",["m1", "c3"], ["cc"], axis=0),
        oh.make_node("Exp",   ["cc"], ["ex"]),
        oh.make_node("Slice", ["ex", "s_starts", "s_ends", "s_axes"], ["sl"]),
        oh.make_node("Mul",   ["sl", "c2"], ["mu"]),
        oh.make_node("LayerNormalization",
                     ["mu", "gamma", "beta"], ["ln"], axis=-1, epsilon=1e-5),
        oh.make_node("Add",   ["ln", "c1"], ["add"]),
        oh.make_node("Relu",  ["add"], ["rl"]),
        oh.make_node("Sub",   ["rl", "c1"], ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "rc07_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, D])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[
            onh.from_array(K, "K"), onh.from_array(tile_rep, "tile_rep"),
            onh.from_array(s_starts, "s_starts"), onh.from_array(s_ends, "s_ends"),
            onh.from_array(s_axes, "s_axes"),
            onh.from_array(c1, "c1"), onh.from_array(c2, "c2"), onh.from_array(c3, "c3"),
            onh.from_array(gamma, "gamma"), onh.from_array(beta, "beta"),
        ],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, D).astype(np.float32)}
    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2
    outs = [(name, run_backend(name, model, inputs)[0])
            for name in ["openvino", "tensorflow", "torch_compile",
                         "torchscript", "xla"]]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
