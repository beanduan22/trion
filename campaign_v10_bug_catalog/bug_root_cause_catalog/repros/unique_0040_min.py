#!/usr/bin/env python3
"""
RC-01 · XLA/TF HLO lowering of attention-style matmul+transpose+softmax.

Cluster: 41 unique bugs.
Representative source: unique_0040 (campaign bug #311).

Trigger pattern (minimal):
    x -> Slice(step=2) -> Tile -> Transpose -> MatMul -> Softmax -> MatMul -> Mul

Reference cluster: onnxruntime, torchscript, torch_compile, tvm, openvino, pytorch_eager.
Suspect cluster  : tensorflow, xla  (both lower through XLA HLO fusion).

Expected behaviour: numeric result matches the reference cluster within 1e-5.
Observed bug      : TF and XLA diverge together by rel_l2 ~ 0.05 – 1.0 depending
                    on shape and attention pattern.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

# ---- tiny shapes ----
B, C, H, W = 1, 8, 8, 8
TOL = 1e-3
np.random.seed(0)


def build_model():
    """Build `Slice -> Tile -> Transpose -> MatMul -> Softmax -> MatMul -> Mul`."""
    # parameters sized to the post-slice/tile shape (last dim stays W)
    WK = np.random.randn(W, W).astype(np.float32)
    WV = np.random.randn(W, W).astype(np.float32)
    scale = np.array([1.0 / np.sqrt(W)], dtype=np.float32)

    nodes = [
        oh.make_node("Slice", ["x", "s_starts", "s_ends", "s_axes", "s_steps"], ["sl"]),
        oh.make_node("Tile",  ["sl", "tile_repeats"],                           ["tl"]),
        oh.make_node("Transpose", ["tl"], ["trans"], perm=[0, 1, 3, 2]),
        oh.make_node("MatMul",    ["trans", "WK"], ["qk"]),
        oh.make_node("Softmax",   ["qk"], ["probs"], axis=-1),
        oh.make_node("MatMul",    ["probs", "WV"], ["av"]),
        oh.make_node("Mul",       ["av", "scale"], ["y"]),
    ]

    graph = oh.make_graph(
        nodes, "rc01_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, C, H, W])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[
            onh.from_array(np.array([0], np.int64),             "s_starts"),
            onh.from_array(np.array([H], np.int64),             "s_ends"),
            onh.from_array(np.array([2], np.int64),             "s_axes"),
            onh.from_array(np.array([2], np.int64),             "s_steps"),
            onh.from_array(np.array([1, 1, 2, 1], np.int64),    "tile_repeats"),
            onh.from_array(WK,    "WK"),
            onh.from_array(WV,    "WV"),
            onh.from_array(scale, "scale"),
        ],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, C, H, W).astype(np.float32)}

    ref, err_ref = run_backend("onnxruntime", model, inputs)
    tf_out, err_tf = run_backend("tensorflow", model, inputs)
    xla_out, err_xla = run_backend("xla", model, inputs)

    if err_ref:
        print(f"[setup] ORT reference failed: {err_ref}"); return 2
    for name, err in [("tensorflow", err_tf), ("xla", err_xla)]:
        if err: print(f"[warn] {name}: {err}")

    code, _ = report(
        expected_pair=("onnxruntime", ref),
        suspect_pair=[("tensorflow", tf_out), ("xla", xla_out)],
        outputs=None, tolerance=TOL,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
