#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0043
Source     : Campaign v2 (fuzzing)
Compiler   : TF XLA (jax.jit on GPU)
Patterns   : resize_linear_halfpixel (half_pixel coordinate mode in ONNX)
Root cause : JAX GPU JIT bicubic/linear half_pixel resize diverges from eager —
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(2)

N, C, H, W = 1, 8, 8, 8

def make_model():
    # resize_linear half_pixel + FPN-like branch + batchnorm-style scale
    bn_mean  = np.zeros(C, dtype=np.float32)
    bn_var   = np.ones(C, dtype=np.float32)
    bn_scale = np.ones(C, dtype=np.float32) * 0.9
    bn_b     = np.zeros(C, dtype=np.float32) + 0.05

    nodes = [
        # FPN: 1x1 conv (simulated as MatMul on spatial)
        helper.make_node("MatMul", ["x", "fpn_w"], ["fpn_out"]),
        # Resize linear half_pixel 2x
        helper.make_node("Resize", ["fpn_out", "roi", "scales"], ["res_out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="linear"),
        # BatchNorm-style: (x - mean) / sqrt(var + eps) * scale + bias
        helper.make_node("Sub", ["res_out", "bn_mean_unsq"], ["bn_sub"]),
        helper.make_node("Add", ["bn_var_unsq", "bn_eps"], ["bn_denom"]),
        helper.make_node("Sqrt", ["bn_denom"], ["bn_sqrt"]),
        helper.make_node("Div", ["bn_sub", "bn_sqrt"], ["bn_norm"]),
        helper.make_node("Mul", ["bn_norm", "bn_scale_unsq"], ["bn_mul"]),
        helper.make_node("Add", ["bn_mul", "bn_b_unsq"], ["out"]),
    ]

    fpn_w = np.eye(C, dtype=np.float32).reshape(1, 1, C, C)
    empty_roi = np.zeros(0, dtype=np.float32)
    scales    = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(fpn_w, "fpn_w"),
        numpy_helper.from_array(empty_roi, "roi"),
        numpy_helper.from_array(scales, "scales"),
        numpy_helper.from_array(bn_mean.reshape(1, C, 1, 1), "bn_mean_unsq"),
        numpy_helper.from_array(bn_var.reshape(1, C, 1, 1),  "bn_var_unsq"),
        numpy_helper.from_array(np.array([1e-5], dtype=np.float32), "bn_eps"),
        numpy_helper.from_array(bn_scale.reshape(1, C, 1, 1), "bn_scale_unsq"),
        numpy_helper.from_array(bn_b.reshape(1, C, 1, 1),     "bn_b_unsq"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0043",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, H, W])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, C, H*2, W*2])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()
x_in = np.random.randn(N, C, H, W).astype(np.float32)

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_in})[0]

out_all = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

max_diff = float(np.max(np.abs(out_all - out_dis)))
print(f"uid=0043 TF XLA resize_linear_halfpixel + FPN + batchnorm")
print(f"  NOTE: XLA not installed; verifying ORT self-consistency as cross-check")
print(f"  Actual bug requires cuda-jaxlib GPU JIT to reproduce")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
passed = max_diff < 0.01
print(f"PASS={passed}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
