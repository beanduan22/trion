#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0055
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : 4D MatMul -> manual LayerNorm -> dilated ASPP -> ConstantOfShape -> Resize(nearest,half_pixel,ceil)
Root cause : ORT nearest-ceil resize with half_pixel coordinate mode after dilated ASPP
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(14)

N, C_in, H, W = 1, 3, 8, 8
C_mid = 4
C_out = 8

def make_model():
    # 4D MatMul: [1,C_in,H,W] @ [1,1,W,W] -> [1,C_in,H,W]; weight [1,1,W,W]
    mm4d_w = np.random.randn(W, W).astype(np.float32) * 0.1
    mm4d_w = mm4d_w.reshape(1, 1, W, W)

    # Manual LayerNorm over last dim (W=8)
    # Note: after 4D matmul, tensor is still [1,C_in,H,W]; LayerNorm last dim = W

    # ASPP: convolutions on [1,C_in,H,W]
    aspp_w1 = np.random.randn(C_mid, C_in, 1, 1).astype(np.float32) * 0.1
    aspp_b1 = np.zeros(C_mid, dtype=np.float32)
    aspp_w2 = np.random.randn(C_mid, C_in, 3, 3).astype(np.float32) * 0.1
    aspp_b2 = np.zeros(C_mid, dtype=np.float32)
    aspp_w3 = np.random.randn(C_mid, C_in, 3, 3).astype(np.float32) * 0.1
    aspp_b3 = np.zeros(C_mid, dtype=np.float32)
    aspp_wm = np.random.randn(C_out, C_mid * 3, 1, 1).astype(np.float32) * 0.1
    aspp_bm = np.zeros(C_out, dtype=np.float32)

    empty_roi = np.zeros(0, dtype=np.float32)
    scales    = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    # ConstantOfShape target: [N, C_out, H, W] (same shape as aspp_out)
    cos_tgt   = np.array([N, C_out, H, W], dtype=np.int64)

    nodes = [
        # 4D MatMul (last-dim mix): [1,C_in,H,W] @ [1,1,W,W] -> [1,C_in,H,W]
        helper.make_node("MatMul", ["x", "mm4d_w"], ["mm_out"]),
        # Manual LayerNorm over last dim (W=8), keepdims=1 to maintain broadcastability
        helper.make_node("ReduceMean", ["mm_out"], ["mln_m"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["mm_out", "mln_m"], ["mln_sub"]),
        helper.make_node("Mul", ["mln_sub", "mln_sub"], ["mln_sq"]),
        helper.make_node("ReduceMean", ["mln_sq"], ["mln_var"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["mln_var", "mln_eps"], ["mln_vadd"]),
        helper.make_node("Sqrt", ["mln_vadd"], ["mln_sqrt"]),
        helper.make_node("Div", ["mln_sub", "mln_sqrt"], ["mln_out"]),
        # ASPP (dilated convolutions on [1,C_in,H,W])
        helper.make_node("Conv", ["mln_out", "aspp_w1", "aspp_b1"], ["aspp_br1"],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]),
        helper.make_node("Conv", ["mln_out", "aspp_w2", "aspp_b2"], ["aspp_br2"],
                         kernel_shape=[3, 3], pads=[2, 2, 2, 2],
                         dilations=[2, 2], strides=[1, 1]),
        helper.make_node("Conv", ["mln_out", "aspp_w3", "aspp_b3"], ["aspp_br3"],
                         kernel_shape=[3, 3], pads=[4, 4, 4, 4],
                         dilations=[4, 4], strides=[1, 1]),
        helper.make_node("Concat", ["aspp_br1", "aspp_br2", "aspp_br3"], ["aspp_cat"],
                         axis=1),
        helper.make_node("Conv", ["aspp_cat", "aspp_wm", "aspp_bm"], ["aspp_out"],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]),
        # ConstantOfShape (zero tensor, same shape) + Add (zero-bias pattern)
        helper.make_node("ConstantOfShape", ["cos_tgt"], ["cos_out"],
                         value=onnx.helper.make_tensor("val", TensorProto.FLOAT, [1], [0.0])),
        helper.make_node("Add", ["aspp_out", "cos_out"], ["cos_add"]),
        # Resize nearest half_pixel ceil 2x
        helper.make_node("Resize", ["cos_add", "roi", "scales"], ["out"],
                         coordinate_transformation_mode="half_pixel",
                         mode="nearest",
                         nearest_mode="ceil"),
    ]

    initializers = [
        numpy_helper.from_array(mm4d_w,   "mm4d_w"),
        numpy_helper.from_array(np.array([1e-5], dtype=np.float32), "mln_eps"),
        numpy_helper.from_array(aspp_w1,  "aspp_w1"),
        numpy_helper.from_array(aspp_b1,  "aspp_b1"),
        numpy_helper.from_array(aspp_w2,  "aspp_w2"),
        numpy_helper.from_array(aspp_b2,  "aspp_b2"),
        numpy_helper.from_array(aspp_w3,  "aspp_w3"),
        numpy_helper.from_array(aspp_b3,  "aspp_b3"),
        numpy_helper.from_array(aspp_wm,  "aspp_wm"),
        numpy_helper.from_array(aspp_bm,  "aspp_bm"),
        numpy_helper.from_array(cos_tgt,   "cos_tgt"),
        numpy_helper.from_array(empty_roi, "roi"),
        numpy_helper.from_array(scales,    "scales"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0055",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C_in, H, W])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, C_out, H*2, W*2])],
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
print(f"uid=0055 Resize(nearest,ceil) + manual LayerNorm + dilated ASPP + ConstantOfShape")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"PASS={passed}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
