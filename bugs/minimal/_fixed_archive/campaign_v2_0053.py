#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0053
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : Abs(x)*x -> Resize(nearest,half_pixel,ceil) -> 4D MatMul -> dilated ASPP conv
Root cause : ORT's nearest-ceil resize with half_pixel coordinate mode may compute
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(12)

N, C_in, H, W = 1, 3, 8, 8
C_mid = 4
C_out = 8
# After 2x resize: [1, C_in, H*2, W*2] = [1, 3, 16, 16]
# 4D MatMul weight: [1, 1, W_res, W_res] where W_res = W*2 = 16
W_res = W * 2

def make_model():
    # 4D MatMul: last dim W_res=16 -> [1,1,16,16]
    mm4d_w = np.eye(W_res, dtype=np.float32).reshape(1, 1, W_res, W_res) * 0.95

    # ASPP: expects [1, C_in=3, H_res=16, W_res=16] as input
    aspp_w1 = np.random.randn(C_mid, C_in, 1, 1).astype(np.float32) * 0.1
    aspp_b1 = np.zeros(C_mid, dtype=np.float32)
    aspp_w2 = np.random.randn(C_mid, C_in, 3, 3).astype(np.float32) * 0.1
    aspp_b2 = np.zeros(C_mid, dtype=np.float32)
    aspp_w3 = np.random.randn(C_mid, C_in, 3, 3).astype(np.float32) * 0.1
    aspp_b3 = np.zeros(C_mid, dtype=np.float32)
    # Merge: 3 * C_mid = 12 input channels -> C_out=8
    aspp_wm = np.random.randn(C_out, C_mid * 3, 1, 1).astype(np.float32) * 0.1
    aspp_bm = np.zeros(C_out, dtype=np.float32)

    empty_roi = np.zeros(0, dtype=np.float32)
    scales    = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # Slice: take [:, :C_out, :H_res, :W_res//2] then pad to match
    H_res = H * 2
    sls = np.array([0, 0, 0, 0], dtype=np.int64)
    sle = np.array([N, C_out, H_res, W_res // 2], dtype=np.int64)
    # Pad to add W_res//2 along last dim: [0,0,0,0, 0,0,0,W_res//2]
    pads_val = np.array([0, 0, 0, 0, 0, 0, 0, W_res // 2], dtype=np.int64)

    nodes = [
        # abs(x) * x  (signed square: keeps sign)
        helper.make_node("Abs", ["x"], ["abs_x"]),
        helper.make_node("Mul", ["abs_x", "x"], ["signed_sq"]),
        # Resize nearest half_pixel ceil 2x
        helper.make_node("Resize", ["signed_sq", "roi", "scales"], ["res_out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="nearest",
                         nearest_mode="ceil"),
        # 4D MatMul: [1,C_in,H_res,W_res] @ [1,1,W_res,W_res] -> [1,C_in,H_res,W_res]
        helper.make_node("MatMul", ["res_out", "mm4d_w"], ["mm_out"]),
        # ASPP branches (conv on [1,C_in,H_res,W_res])
        helper.make_node("Conv", ["mm_out", "aspp_w1", "aspp_b1"], ["aspp_br1"],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]),
        helper.make_node("Conv", ["mm_out", "aspp_w2", "aspp_b2"], ["aspp_br2"],
                         kernel_shape=[3, 3], pads=[2, 2, 2, 2],
                         dilations=[2, 2], strides=[1, 1]),
        helper.make_node("Conv", ["mm_out", "aspp_w3", "aspp_b3"], ["aspp_br3"],
                         kernel_shape=[3, 3], pads=[4, 4, 4, 4],
                         dilations=[4, 4], strides=[1, 1]),
        helper.make_node("Concat", ["aspp_br1", "aspp_br2", "aspp_br3"], ["aspp_cat"],
                         axis=1),
        helper.make_node("Conv", ["aspp_cat", "aspp_wm", "aspp_bm"], ["aspp_out"],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]),
        # Slice + Pad pattern (spatial crop)
        helper.make_node("Slice", ["aspp_out", "sls", "sle"], ["sp_sl"]),
        helper.make_node("Pad",   ["sp_sl", "pads_val"], ["sp_pd"], mode="constant"),
        helper.make_node("Concat", ["aspp_out", "sp_pd"], ["out"], axis=1),
    ]

    initializers = [
        numpy_helper.from_array(empty_roi, "roi"),
        numpy_helper.from_array(scales,    "scales"),
        numpy_helper.from_array(mm4d_w,   "mm4d_w"),
        numpy_helper.from_array(aspp_w1,  "aspp_w1"),
        numpy_helper.from_array(aspp_b1,  "aspp_b1"),
        numpy_helper.from_array(aspp_w2,  "aspp_w2"),
        numpy_helper.from_array(aspp_b2,  "aspp_b2"),
        numpy_helper.from_array(aspp_w3,  "aspp_w3"),
        numpy_helper.from_array(aspp_b3,  "aspp_b3"),
        numpy_helper.from_array(aspp_wm,  "aspp_wm"),
        numpy_helper.from_array(aspp_bm,  "aspp_bm"),
        numpy_helper.from_array(sls,       "sls"),
        numpy_helper.from_array(sle,       "sle"),
        numpy_helper.from_array(pads_val,  "pads_val"),
    ]

    out_C = C_out * 2  # concat of aspp_out and sp_pd (both C_out channels)
    graph = helper.make_graph(
        nodes, "bug_v2_0053",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C_in, H, W])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, out_C, H_res, W_res])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()
x_in = np.random.randn(N, C_in, H, W).astype(np.float32)

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_in})[0]

out_all = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

max_diff = float(np.max(np.abs(out_all - out_dis)))
passed = max_diff < 0.01
print(f"uid=0053 Resize(nearest,half_pixel,ceil) + abs(x)*x + dilated ASPP")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"PASS={passed}")

PASS = passed
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
