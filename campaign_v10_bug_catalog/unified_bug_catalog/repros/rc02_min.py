#!/usr/bin/env python3
"""
RC-02 · XLA + Inductor Conv+BN+Relu fusion divergence.

Cluster: 26 unique bugs.
Representative source: unique_0064 (campaign bug #394).

Trigger pattern (minimal):
    x -> MatMul(q) ; Relu ; Softmax ; MatMul
      -> Conv -> BatchNormalization -> Relu
    +broadcast 1x1x1x1 constant Add upstream.

Reference cluster: onnxruntime, torchscript, tvm, openvino, pytorch_eager.
Suspect cluster  : tensorflow, torch_compile (Inductor), xla.

These three "aggressive optimizer" backends fold BN into Conv weights/biases
before execution. When an extra element-wise broadcast sits upstream of the
Conv, the folding commutation loses precision in fp32 and the three backends
agree on a wrong answer.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, Cin, H, W = 1, 8, 8, 8
Cout = 8
TOL = 1e-3
np.random.seed(1)


def build_model():
    bc_add = np.array([[[[0.01]]]], dtype=np.float32)     # 1x1x1x1 broadcast

    conv_w = np.random.randn(Cout, Cin, 3, 3).astype(np.float32) * 0.1
    conv_b = np.random.randn(Cout).astype(np.float32) * 0.01
    bn_scale = np.ones(Cout, dtype=np.float32) * 0.9
    bn_bias  = np.zeros(Cout, dtype=np.float32)
    bn_mean  = np.zeros(Cout, dtype=np.float32)
    bn_var   = np.ones(Cout, dtype=np.float32)

    nodes = [
        oh.make_node("Add",  ["x", "bc_add"],                              ["xa"]),
        oh.make_node("Conv", ["xa", "conv_w", "conv_b"],                   ["c"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        oh.make_node("BatchNormalization",
                     ["c", "bn_scale", "bn_bias", "bn_mean", "bn_var"],    ["bn"],
                     epsilon=1e-5),
        oh.make_node("Relu", ["bn"], ["y"]),
    ]

    graph = oh.make_graph(
        nodes, "rc02_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, Cin, H, W])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[
            onh.from_array(bc_add,   "bc_add"),
            onh.from_array(conv_w,   "conv_w"),
            onh.from_array(conv_b,   "conv_b"),
            onh.from_array(bn_scale, "bn_scale"),
            onh.from_array(bn_bias,  "bn_bias"),
            onh.from_array(bn_mean,  "bn_mean"),
            onh.from_array(bn_var,   "bn_var"),
        ],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(B, Cin, H, W).astype(np.float32)}

    ref, err_ref = run_backend("onnxruntime", model, inputs)
    tf_out,  _ = run_backend("tensorflow",    model, inputs)
    tc_out,  _ = run_backend("torch_compile", model, inputs)
    xla_out, _ = run_backend("xla",           model, inputs)

    if err_ref:
        print(f"[setup] ORT reference failed: {err_ref}"); return 2

    code, _ = report(
        expected_pair=("onnxruntime", ref),
        suspect_pair=[("tensorflow", tf_out),
                      ("torch_compile", tc_out),
                      ("xla", xla_out)],
        outputs=None, tolerance=TOL,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
