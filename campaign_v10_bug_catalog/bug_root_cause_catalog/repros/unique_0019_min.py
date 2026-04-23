#!/usr/bin/env python3
"""
RC-05 · TF graph-mode Softmax(axis=0) inside Greater+Where mask.

Cluster: 5 unique bugs.
Representative source: unique_0019 (campaign bug #1095).

Trigger pattern:
    x -> MatMul -> MatMul -> MatMul -> Softmax(axis=0)
      -> Add(residual) -> ReduceMean -> Greater(const) -> Where(mask)
      -> Softmax -> MatMul

Reference cluster: every backend except tensorflow (including XLA — so it is
NOT a shared XLA-HLO issue).
Suspect          : tensorflow only.

TF's graph executor reorders the Softmax reduction when the Softmax axis is
not the last axis AND the downstream op is a Greater/Where masked gather.
XLA bypasses TF's reorder, so TF-XLA matches the reference cluster.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

N = 8
D = 8
TOL = 1e-3
np.random.seed(4)


def build_model():
    W1 = np.random.randn(D, D).astype(np.float32)
    W2 = np.random.randn(D, D).astype(np.float32)
    W3 = np.random.randn(D, D).astype(np.float32)
    W4 = np.random.randn(D, D).astype(np.float32)
    thr = np.array([0.0], dtype=np.float32)
    mask_fill = np.array([-1e4], dtype=np.float32)
    axes_last = np.array([-1], dtype=np.int64)

    nodes = [
        oh.make_node("MatMul",  ["x", "W1"], ["m1"]),
        oh.make_node("MatMul",  ["m1", "W2"], ["m2"]),
        oh.make_node("MatMul",  ["m2", "W3"], ["m3"]),
        oh.make_node("Softmax", ["m3"], ["sm0"], axis=0),                 # softmax on first axis
        oh.make_node("Add",     ["sm0", "x"], ["resid"]),
        oh.make_node("ReduceMean", ["resid", "axes_last"], ["rm"], keepdims=1),
        oh.make_node("Greater", ["rm", "thr"], ["mask_b"]),
        oh.make_node("Where",   ["mask_b", "resid", "mask_fill"], ["masked"]),
        oh.make_node("Softmax", ["masked"], ["sm1"], axis=-1),
        oh.make_node("MatMul",  ["sm1", "W4"], ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "rc05_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [N, D])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[onh.from_array(W1, "W1"), onh.from_array(W2, "W2"),
                     onh.from_array(W3, "W3"), onh.from_array(W4, "W4"),
                     onh.from_array(thr, "thr"),
                     onh.from_array(mask_fill, "mask_fill"),
                     onh.from_array(axes_last, "axes_last")],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(N, D).astype(np.float32)}

    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2
    tf, _ = run_backend("tensorflow", model, inputs)
    xla, _ = run_backend("xla", model, inputs)
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=[("tensorflow", tf), ("xla", xla)],
                     outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
