#!/usr/bin/env python3
"""
RC-06 · ONNX-spec ambiguity: Squeeze/Unsqueeze + bias-Softmax + LayerNorm.

Cluster: 5 unique bugs.
Representative source: unique_0095 (campaign bug #253).

Trigger pattern:
    x -> Add -> Add -> Add (triple residual)
      -> Unsqueeze(axes=[2]) -> Squeeze(axes=[2])
      -> Add(bias) -> Softmax -> Sigmoid -> Mul
      -> Add -> LayerNormalization

Reference cluster: pytorch_eager + onnxruntime ONLY.
Suspect          : every other backend (openvino, tensorflow, torch_compile,
                   torchscript, tvm, xla) — each gives a different answer.

ONNX opset-13 moved the `axes` argument of Squeeze/Unsqueeze from attribute
to input.  Different compiler frontends handle the round-trip
Unsqueeze([2]) -> Squeeze([2]) inconsistently when combined with an
immediately-downstream bias+Softmax fusion: TS folds it to identity, OV
keeps both ops, TVM re-orders, XLA/Inductor hoist the bias past the
reshape.  Only ORT and onnx2torch-eager preserve the literal spec.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, N, D = 1, 8, 16
TOL = 1e-3
np.random.seed(5)


def build_model():
    b1 = np.random.randn(1, 1, D).astype(np.float32) * 0.1
    b2 = np.random.randn(1, 1, D).astype(np.float32) * 0.1
    b3 = np.random.randn(1, 1, D).astype(np.float32) * 0.1
    bias = np.random.randn(1, 1, D).astype(np.float32) * 0.1
    gamma = np.ones(D, dtype=np.float32)
    beta  = np.zeros(D, dtype=np.float32)
    ax_u = np.array([2], dtype=np.int64)
    ax_s = np.array([2], dtype=np.int64)

    nodes = [
        oh.make_node("Add", ["x",   "b1"], ["a1"]),
        oh.make_node("Add", ["a1",  "b2"], ["a2"]),
        oh.make_node("Add", ["a2",  "b3"], ["a3"]),
        oh.make_node("Unsqueeze", ["a3", "ax_u"], ["uns"]),
        oh.make_node("Squeeze",   ["uns", "ax_s"], ["sq"]),
        oh.make_node("Add",     ["sq", "bias"], ["sb"]),
        oh.make_node("Softmax", ["sb"], ["sm"], axis=-1),
        oh.make_node("Sigmoid", ["sm"], ["sig"]),
        oh.make_node("Mul",     ["sig", "sb"], ["mul"]),
        oh.make_node("Add",     ["mul", "x"], ["res"]),
        oh.make_node("LayerNormalization",
                     ["res", "gamma", "beta"], ["y"], axis=-1, epsilon=1e-5),
    ]
    graph = oh.make_graph(
        nodes, "rc06_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, N, D])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[onh.from_array(b1, "b1"), onh.from_array(b2, "b2"),
                     onh.from_array(b3, "b3"), onh.from_array(bias, "bias"),
                     onh.from_array(gamma, "gamma"), onh.from_array(beta, "beta"),
                     onh.from_array(ax_u, "ax_u"),   onh.from_array(ax_s, "ax_s")],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, N, D).astype(np.float32)}

    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2

    outs = [(name, run_backend(name, model, inputs)[0])
            for name in ["openvino", "tensorflow", "torch_compile",
                         "torchscript", "tvm", "xla"]]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
