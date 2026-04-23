#!/usr/bin/env python3
"""
RC-04 · Inductor-only Conv -> Relu -> Softmax miscompile.

Cluster: 5 unique bugs.
Representative source: unique_0034 (campaign bug #1177).

Trigger pattern:
    x -> Conv -> Relu -> Softmax(axis=1)
       -> Expand(broadcast) -> Add
       -> Conv -> Relu (repeat)

Reference cluster: every backend except torch.compile.
Suspect          : torch_compile (Inductor) only.

Inductor vertically fuses Conv+Relu+Softmax into one Triton kernel; the
per-tile partial-sum for Softmax is computed per-tile rather than per-row,
so the normalization factor is wrong.  The error is bounded but clearly
above tolerance for any non-trivial channel dim.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, Cin, H, W = 1, 4, 8, 8
Cout = 8
TOL = 1e-3
np.random.seed(3)


def build_model():
    conv_w1 = np.random.randn(Cout, Cin,  3, 3).astype(np.float32) * 0.1
    conv_b1 = np.zeros(Cout, dtype=np.float32)

    conv_w2 = np.random.randn(Cout, Cout, 3, 3).astype(np.float32) * 0.1
    conv_b2 = np.zeros(Cout, dtype=np.float32)

    bias = np.random.randn(1, Cout, 1, 1).astype(np.float32) * 0.01

    nodes = [
        oh.make_node("Conv",    ["x", "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        oh.make_node("Relu",    ["c1"], ["r1"]),
        oh.make_node("Softmax", ["r1"], ["s1"], axis=1),                    # channel-softmax
        oh.make_node("Add",     ["s1", "bias"], ["add"]),
        oh.make_node("Conv",    ["add", "w2", "b2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        oh.make_node("Relu",    ["c2"], ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "rc04_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, Cin, H, W])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[
            onh.from_array(conv_w1, "w1"), onh.from_array(conv_b1, "b1"),
            onh.from_array(conv_w2, "w2"), onh.from_array(conv_b2, "b2"),
            onh.from_array(bias,    "bias"),
        ],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, Cin, H, W).astype(np.float32)}

    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2
    tc, err_tc = run_backend("torch_compile", model, inputs)
    if err_tc: print(f"[warn] torch_compile: {err_tc}")
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=[("torch_compile", tc)],
                     outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
