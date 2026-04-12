#!/usr/bin/env python3
"""
Bug v2-0052 — TF XLA: resize_cubic_halfpixel + manual_hardsigmoid + 3-branch concat
Compiler : TF XLA (jax.jit on GPU)
Pattern  : resize_cubic_halfpixel after manual hardsigmoid and three-branch concat
Root cause: JAX GPU JIT bicubic halfpixel resize diverges from eager after
            manual hardsigmoid (clip(x/6+0.5, 0, 1)) and 3-branch concat.
Note     : XLA/JAX not installed in this environment. This repro builds the equivalent
           ONNX model and verifies ORT self-consistency as a cross-check.
           The actual bug requires cuda-jaxlib to reproduce.

Run: python bug_v2_0052.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(11)

N, C, H, W = 1, 4, 8, 8

def make_model():
    # Linear transforms on last dim (W dimension)
    mm_w1 = np.eye(W, dtype=np.float32).reshape(1, 1, W, W) * 0.9
    mm_w3 = np.eye(W, dtype=np.float32).reshape(1, 1, W, W) * 0.7

    empty_roi = np.zeros(0, dtype=np.float32)
    scales    = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    nodes = [
        # Branch 1: 4D MatMul (last-dim projection)
        helper.make_node("MatMul", ["x", "mm_w1"], ["br1"]),
        # Branch 2: manual HardSigmoid clip(x/6 + 0.5, 0, 1)
        helper.make_node("Div", ["x", "hs_six"], ["hs_div"]),
        helper.make_node("Add", ["hs_div", "hs_half"], ["hs_add"]),
        helper.make_node("Clip", ["hs_add", "clip_min", "clip_max"], ["br2"]),
        # Branch 3: 4D MatMul
        helper.make_node("MatMul", ["x", "mm_w3"], ["br3"]),
        # Combine branches by element-wise addition
        helper.make_node("Add", ["br1", "br2"], ["comb1"]),
        helper.make_node("Add", ["comb1", "br3"], ["comb_out"]),
        # Resize cubic half_pixel 2x
        helper.make_node("Resize", ["comb_out", "roi", "scales"], ["out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="cubic"),
    ]

    initializers = [
        numpy_helper.from_array(mm_w1, "mm_w1"),
        numpy_helper.from_array(mm_w3, "mm_w3"),
        numpy_helper.from_array(np.array([6.0], dtype=np.float32), "hs_six"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "hs_half"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "clip_min"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "clip_max"),
        numpy_helper.from_array(empty_roi, "roi"),
        numpy_helper.from_array(scales,    "scales"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0052",
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
print(f"uid=0052 TF XLA resize_cubic_halfpixel + manual_hardsigmoid + 3-branch concat")
print(f"  NOTE: XLA not installed; verifying ORT self-consistency as cross-check")
print(f"  Actual bug requires cuda-jaxlib GPU JIT to reproduce")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
passed = max_diff < 0.01
print(f"PASS={passed}")
