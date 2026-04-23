#!/usr/bin/env python3
"""
RC-08 · Depthwise Conv + BN fold + Resize-nearest-asymmetric.

Cluster: 3 unique bugs.
Representative source: unique_0088 (campaign bug #1136).

Trigger pattern:
    x -> Add -> Relu -> Conv(group=C, depthwise)
      -> Add -> Mul -> Cast(fp32->fp16)->Cast(fp16->fp32)
      -> Mul -> Conv -> Resize(nearest, asymmetric) x2 -> Conv
      -> LayerNormalization

Reference cluster: pytorch_eager + onnxruntime + tensorflow.
Suspect          : openvino, torch_compile, torchscript, tvm, xla.

Two interacting issues co-occur:
  (a) depthwise Conv + BN folding: TF + ORT + eager preserve per-channel
      scale/bias in order; the five suspects commute the fold differently.
  (b) Resize nearest-asymmetric with `round_prefer_floor` is spec-ambiguous.

rel_l2 ranges 0.06 – 1.0.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, C, H, W = 1, 8, 8, 8
TOL = 1e-3
np.random.seed(8)


def build_model():
    # depthwise conv: groups=C, out_channels=C
    dw_w = np.random.randn(C, 1, 3, 3).astype(np.float32) * 0.1
    dw_b = np.zeros(C, dtype=np.float32)

    conv_w = np.random.randn(C, C, 1, 1).astype(np.float32) * 0.1
    conv_b = np.zeros(C, dtype=np.float32)

    bias  = np.random.randn(1, C, 1, 1).astype(np.float32) * 0.01
    scale = np.ones((1, C, 1, 1), dtype=np.float32) * 0.9

    sizes = np.array([B, C, 2 * H, 2 * W], dtype=np.int64)
    scales_empty = np.array([], dtype=np.float32)
    roi_empty = np.array([], dtype=np.float32)
    # LayerNorm axis=1 means normalized shape is X.shape[1:]
    ln_norm_shape = (C, 2 * H, 2 * W)
    gamma = np.ones(ln_norm_shape, dtype=np.float32)
    beta = np.zeros(ln_norm_shape, dtype=np.float32)

    nodes = [
        oh.make_node("Add",  ["x", "bias"], ["a0"]),
        oh.make_node("Relu", ["a0"], ["r0"]),
        oh.make_node("Conv", ["r0", "dw_w", "dw_b"], ["dw"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=C),
        oh.make_node("Add",  ["dw", "bias"],  ["a1"]),
        oh.make_node("Mul",  ["a1", "scale"], ["m1"]),
        oh.make_node("Cast", ["m1"],  ["m16"], to=TP.FLOAT16),
        oh.make_node("Cast", ["m16"], ["mf"],  to=TP.FLOAT),
        oh.make_node("Conv", ["mf", "conv_w", "conv_b"], ["c1"],
                     kernel_shape=[1, 1]),
        oh.make_node("Resize",
                     ["c1", "roi_empty", "scales_empty", "sizes"],
                     ["rz"],
                     mode="nearest", coordinate_transformation_mode="asymmetric",
                     nearest_mode="round_prefer_floor"),
        oh.make_node("Conv", ["rz", "conv_w", "conv_b"], ["c2"],
                     kernel_shape=[1, 1]),
        oh.make_node("LayerNormalization",
                     ["c2", "gamma", "beta"], ["y"], axis=1, epsilon=1e-5),
    ]
    graph = oh.make_graph(
        nodes, "rc08_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, C, H, W])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[
            onh.from_array(dw_w, "dw_w"),   onh.from_array(dw_b, "dw_b"),
            onh.from_array(conv_w, "conv_w"), onh.from_array(conv_b, "conv_b"),
            onh.from_array(bias, "bias"),   onh.from_array(scale, "scale"),
            onh.from_array(sizes, "sizes"),
            onh.from_array(scales_empty, "scales_empty"),
            onh.from_array(roi_empty, "roi_empty"),
            onh.from_array(gamma, "gamma"), onh.from_array(beta, "beta"),
        ],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, C, H, W).astype(np.float32)}
    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2
    outs = [(name, run_backend(name, model, inputs)[0])
            for name in ["openvino", "torchscript", "torch_compile", "tvm", "xla"]]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
