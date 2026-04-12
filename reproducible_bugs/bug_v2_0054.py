#!/usr/bin/env python3
"""
Bug v2-0054 — ORT: Resize(linear, asymmetric) + back-to-back 4D MatMul + reshape
Compiler : OnnxRuntime
Pattern  : Resize(linear, asymmetric) -> 4D MatMul -> 4D MatMul -> Reshape -> Reshape -> Sub -> Div
Root cause: ORT may merge the back-to-back 4D MatMuls (both 1x1xCxC identity-like)
            with the linear asymmetric Resize, or apply incorrect constant folding
            on the reshape chain after the double matmul.

Run: python bug_v2_0054.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(13)

N, C, H, W = 1, 3, 8, 8
# After 2x resize: [1, C, H*2, W*2] = [1, 3, 16, 16]
# 4D MatMul with weight [1, 1, W_res, W_res] where W_res=16
W_res = W * 2  # 16

def make_model():
    mm4d_w1 = np.eye(W_res, dtype=np.float32).reshape(1, 1, W_res, W_res) * 0.95
    mm4d_w2 = np.eye(W_res, dtype=np.float32).reshape(1, 1, W_res, W_res) * 0.90

    empty_roi  = np.zeros(0, dtype=np.float32)
    scales_lin = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # Reshape: [1, C, H_res, W_res] -> [1, -1] -> [1, C, H_res, W_res]
    H_res = H * 2
    flat_shape = np.array([1, -1], dtype=np.int64)
    back_shape = np.array([N, C, H_res, W_res], dtype=np.int64)

    nodes = [
        # Resize linear asymmetric 2x: [1,C,H,W] -> [1,C,H_res,W_res]
        helper.make_node("Resize", ["x", "roi", "scales"], ["res_out"],
                         coordinate_transformation_mode="asymmetric",
                         mode="linear"),
        # Back-to-back 4D MatMul: [1,C,H_res,W_res] @ [1,1,W_res,W_res]
        helper.make_node("MatMul", ["res_out", "mm4d_w1"], ["mm1_out"]),
        helper.make_node("MatMul", ["mm1_out",  "mm4d_w2"], ["mm2_out"]),
        # Reshape flatten then restore (identity in shape)
        helper.make_node("Reshape", ["mm2_out", "flat_shape"], ["rf_flat"]),
        helper.make_node("Reshape", ["rf_flat",  "back_shape"], ["rf_out"]),
        # Sub zero, div one (identity operations to exercise constant folding)
        helper.make_node("Sub", ["rf_out", "sub_z"], ["sub_out"]),
        helper.make_node("Div", ["sub_out", "div_o"], ["out"]),
    ]

    initializers = [
        numpy_helper.from_array(empty_roi,  "roi"),
        numpy_helper.from_array(scales_lin, "scales"),
        numpy_helper.from_array(mm4d_w1,    "mm4d_w1"),
        numpy_helper.from_array(mm4d_w2,    "mm4d_w2"),
        numpy_helper.from_array(flat_shape,  "flat_shape"),
        numpy_helper.from_array(back_shape,  "back_shape"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "sub_z"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "div_o"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0054",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, H, W])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, C, H_res, W_res])],
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

max_diff_ort = float(np.max(np.abs(out_all - out_dis)))

print(f"uid=0054 Resize(linear,asymmetric) + back-to-back 4D MatMul")
print(f"  ORT_ENABLE_ALL  vs ORT_DISABLE_ALL max_diff={max_diff_ort:.6f}")
passed = max_diff_ort < 0.01
print(f"PASS={passed}")
