#!/usr/bin/env python3
"""
Bug v2-0041 — ORT: triple-resize chain (asymmetric/nearest -> half_pixel/cubic -> half_pixel/nearest/ceil)
Compiler : OnnxRuntime
Pattern  : triple-resize chain + AdaLayerNorm
Root cause: ORT optimizer may merge or mis-order resize ops when different coordinate
            transformation modes are used in sequence, producing wrong output values.

Run: python bug_v2_0041.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(0)

# Input: [1, 3, 8, 8] NCHW. LayerNorm over last dim (W=8), scales [8].
# After 3x 2x resizes: [1, 3, 64, 64]
N, C, H, W = 1, 3, 8, 8

ln_sc  = np.ones(W, dtype=np.float32)          # last dim = W
ln_b   = np.zeros(W, dtype=np.float32)
ada_sc = np.ones((1, 1, 1, W), dtype=np.float32) * 1.1
ada_b  = np.zeros((1, 1, 1, W), dtype=np.float32) + 0.05
# FiLM: per-channel scale/bias applied after 2x resize -> [1,3,32,32]
gamma  = np.ones((1, C, 1, 1), dtype=np.float32) * 0.9
beta   = np.zeros((1, C, 1, 1), dtype=np.float32) + 0.1
# Residual at scale after 2 resizes [1,3,32,32]
res    = np.zeros((1, C, H*4, W*4), dtype=np.float32)

def make_model():
    nodes = [
        # AdaLayerNorm: LayerNorm(last axis) then affine scale/shift
        helper.make_node("LayerNormalization", ["x", "ln_sc", "ln_b"], ["ln_out"],
                         axis=-1, epsilon=1e-5),
        helper.make_node("Mul", ["ln_out", "ada_sc"], ["ada_mul"]),
        helper.make_node("Add", ["ada_mul", "ada_b"], ["ada_out"]),
        # Resize 1: asymmetric + nearest/floor  x2
        helper.make_node("Resize",
                         ["ada_out", "roi1", "scales1"],
                         ["resize1_out"],
                         coordinate_transformation_mode="asymmetric",
                         mode="nearest",
                         nearest_mode="floor"),
        # Resize 2: half_pixel + cubic  x2  -> [1,3,16,16] after two 2x resizes
        helper.make_node("Resize",
                         ["resize1_out", "roi2", "scales2"],
                         ["resize2_out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="cubic"),
        # FiLM conditioning + residual
        helper.make_node("Mul", ["resize2_out", "gamma"], ["film_mul"]),
        helper.make_node("Add", ["film_mul", "beta"], ["film_add"]),
        helper.make_node("Add", ["film_add", "res"], ["film_out"]),
        # Resize 3: half_pixel + nearest/ceil  x2  -> [1,3,32,32]
        helper.make_node("Resize",
                         ["film_out", "roi3", "scales3"],
                         ["out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="nearest",
                         nearest_mode="ceil"),
    ]

    empty_roi = np.zeros(0, dtype=np.float32)
    scales_2x = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(ln_sc,  "ln_sc"),
        numpy_helper.from_array(ln_b,   "ln_b"),
        numpy_helper.from_array(ada_sc, "ada_sc"),
        numpy_helper.from_array(ada_b,  "ada_b"),
        numpy_helper.from_array(empty_roi,   "roi1"),
        numpy_helper.from_array(scales_2x,   "scales1"),
        numpy_helper.from_array(empty_roi,   "roi2"),
        numpy_helper.from_array(scales_2x,   "scales2"),
        numpy_helper.from_array(gamma,  "gamma"),
        numpy_helper.from_array(beta,   "beta"),
        numpy_helper.from_array(res,    "res"),
        numpy_helper.from_array(empty_roi,   "roi3"),
        numpy_helper.from_array(scales_2x,   "scales3"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0041",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, H, W])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, C, H*8, W*8])],
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
passed = max_diff < 0.01
print(f"uid=0041 triple-resize chain (asymmetric/nearest -> half_pixel/cubic -> half_pixel/nearest/ceil)")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"PASS={passed}")
