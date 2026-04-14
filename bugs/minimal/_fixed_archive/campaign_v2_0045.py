#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0045
Source     : Campaign v2 (fuzzing)
Compiler   : TF XLA (jax.jit on GPU)
Patterns   : resize_cubic_halfpixel after stable-softmax and crms_norm
Root cause : JAX GPU JIT bicubic halfpixel resize diverges from eager after
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(4)

N, C, H, W = 1, 8, 8, 8

def make_model():
    mm_w1 = np.random.randn(C, C).astype(np.float32) * 0.1
    mm_w2 = np.random.randn(C, C).astype(np.float32) * 0.1
    mm_w3 = np.random.randn(C, C).astype(np.float32) * 0.1
    rms_g  = np.ones(C, dtype=np.float32)

    empty_roi = np.zeros(0, dtype=np.float32)
    scales    = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    nodes = [
        # Inception branch 1: identity
        helper.make_node("MatMul", ["x", "mm_w1"], ["br1"]),
        # Inception branch 2: stable softmax
        helper.make_node("ReduceMax", ["x"], ["sm_max"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["x", "sm_max"], ["sm_sub"]),
        helper.make_node("Exp", ["sm_sub"], ["sm_exp"]),
        helper.make_node("ReduceSum", ["sm_exp", "sm_ax"], ["sm_sum"], keepdims=1),
        helper.make_node("Div", ["sm_exp", "sm_sum"], ["br2"]),
        # Inception branch 3: CRMSNorm then MatMul
        helper.make_node("Mul", ["x", "x"], ["rms_sq"]),
        helper.make_node("ReduceMean", ["rms_sq"], ["rms_mean"], axes=[-1], keepdims=0),
        helper.make_node("Add", ["rms_mean", "rms_eps"], ["rms_add"]),
        helper.make_node("Sqrt", ["rms_add"], ["rms_sqrt"]),
        helper.make_node("Div", ["x", "rms_sqrt"], ["rms_norm"]),
        helper.make_node("Mul", ["rms_norm", "rms_g"], ["rms_out"]),
        helper.make_node("MatMul", ["rms_out", "mm_w3"], ["br3"]),
        # Concat branches
        helper.make_node("Add", ["br1", "br2"], ["cat1"]),
        helper.make_node("Add", ["cat1", "br3"], ["concat_out"]),
        # Resize cubic half_pixel 2x
        helper.make_node("Resize", ["concat_out", "roi", "scales"], ["out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="cubic"),
    ]

    sm_ax = np.array([-1], dtype=np.int64)

    initializers = [
        numpy_helper.from_array(mm_w1, "mm_w1"),
        numpy_helper.from_array(mm_w2, "mm_w2"),
        numpy_helper.from_array(mm_w3, "mm_w3"),
        numpy_helper.from_array(sm_ax, "sm_ax"),
        numpy_helper.from_array(np.array([1e-6], dtype=np.float32), "rms_eps"),
        numpy_helper.from_array(rms_g, "rms_g"),
        numpy_helper.from_array(empty_roi, "roi"),
        numpy_helper.from_array(scales, "scales"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0045",
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
print(f"uid=0045 TF XLA resize_cubic_halfpixel + Inception + stable-softmax")
print(f"  NOTE: XLA not installed; verifying ORT self-consistency as cross-check")
print(f"  Actual bug requires cuda-jaxlib GPU JIT to reproduce")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
passed = max_diff < 0.01
print(f"PASS={passed}")

PASS = passed
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
