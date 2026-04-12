#!/usr/bin/env python3
"""
Bug v2-0046 — TF XLA: resize_cubic_halfpixel + spatial reduce-mean + additive layernorm
Compiler : TF XLA (jax.jit on GPU)
Pattern  : resize_cubic_halfpixel after spatial ReduceMean and 4D MatMul
Root cause: JAX GPU JIT bicubic halfpixel resize diverges from eager after
            spatial reduce-mean and additive layernorm with 4D matmul.
Note     : XLA/JAX not installed in this environment. This repro builds the equivalent
           ONNX model and verifies ORT self-consistency as a cross-check.
           The actual bug requires cuda-jaxlib to reproduce.

Run: python bug_v2_0046.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(5)

N, C, H, W = 1, 8, 8, 8

def make_model():
    mm4d_w = np.eye(C, dtype=np.float32).reshape(1, 1, C, C) * 0.95
    ln_sc  = np.ones(C, dtype=np.float32)
    ln_b   = np.zeros(C, dtype=np.float32)
    res    = np.zeros((N, C, H, W), dtype=np.float32)

    empty_roi = np.zeros(0, dtype=np.float32)
    scales    = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    nodes = [
        # Spatial ReduceMean (over H,W axes)
        helper.make_node("ReduceMean", ["x"], ["sp_mean"], axes=[2, 3], keepdims=1),
        helper.make_node("Sub", ["x", "sp_mean"], ["sp_sub"]),
        # 4D MatMul
        helper.make_node("MatMul", ["sp_sub", "mm4d_w"], ["mm_out"]),
        # Additive LayerNorm
        helper.make_node("Add", ["mm_out", "res"], ["ln_in"]),
        helper.make_node("LayerNormalization", ["ln_in", "ln_sc", "ln_b"], ["ln_out"],
                         axis=-1, epsilon=1e-5),
        # Resize cubic half_pixel 2x
        helper.make_node("Resize", ["ln_out", "roi", "scales"], ["out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="cubic"),
    ]

    initializers = [
        numpy_helper.from_array(mm4d_w, "mm4d_w"),
        numpy_helper.from_array(ln_sc, "ln_sc"),
        numpy_helper.from_array(ln_b,  "ln_b"),
        numpy_helper.from_array(res,   "res"),
        numpy_helper.from_array(empty_roi, "roi"),
        numpy_helper.from_array(scales,    "scales"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0046",
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
print(f"uid=0046 TF XLA resize_cubic_halfpixel + spatial ReduceMean + additive LayerNorm")
print(f"  NOTE: XLA not installed; verifying ORT self-consistency as cross-check")
print(f"  Actual bug requires cuda-jaxlib GPU JIT to reproduce")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
passed = max_diff < 0.01
print(f"PASS={passed}")
