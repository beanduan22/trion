#!/usr/bin/env python3
"""
Bug v2-0044 — ORT: Resize(linear, asymmetric) after SE-block sigmoid fusion
Compiler : OnnxRuntime
Pattern  : Resize(linear, asymmetric) + SE-block (GlobalAvgPool->FC->Sigmoid) +
           LayerNorm + Dropout + 4D MatMul + Resize(nearest, asymmetric)
Root cause: ORT fuses Sigmoid inside the SE block and may also apply Resize fusion;
            the combination of linear asymmetric Resize with SE-block sigmoid fusion
            and a second nearest-floor asymmetric Resize can produce wrong results
            when ORT_ENABLE_ALL is active.

Run: python bug_v2_0044.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(3)

N, C, H, W = 1, 8, 8, 8
C_red = 4  # SE reduction ratio

def make_model():
    se_w1   = np.random.randn(C, C_red).astype(np.float32) * 0.1
    se_b1   = np.zeros(C_red, dtype=np.float32)
    se_w2   = np.random.randn(C_red, C).astype(np.float32) * 0.1
    se_b2   = np.zeros(C, dtype=np.float32)
    ln_sc   = np.ones(C, dtype=np.float32)
    ln_b    = np.zeros(C, dtype=np.float32)
    # 4D MatMul: weight [1,1,W_after_resize, W_after_resize]
    # After first linear 2x resize: [1,C,16,16], last dim = 16
    W_res   = W * 2  # 16
    mm4d_w  = np.eye(W_res, dtype=np.float32).reshape(1, 1, W_res, W_res) * 0.9

    empty_roi  = np.zeros(0, dtype=np.float32)
    scales_lin = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    scales_nst = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    # SE reshape: [1,C] -> [1,C,1,1]
    rs_shape   = np.array([1, C, 1, 1], dtype=np.int64)

    nodes = [
        # Resize linear asymmetric 2x: [1,C,H,W] -> [1,C,16,16]
        helper.make_node("Resize", ["x", "roi_lin", "scales_lin"], ["res1_out"],
                         coordinate_transformation_mode="asymmetric",
                         mode="linear"),
        # SE block: GlobalAvgPool -> Flatten -> FC -> ReLU -> FC -> Sigmoid -> Reshape -> Mul
        helper.make_node("GlobalAveragePool", ["res1_out"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "se_w1"], ["se_mm1"]),
        helper.make_node("Add", ["se_mm1", "se_b1"], ["se_add1"]),
        helper.make_node("Relu", ["se_add1"], ["se_relu"]),
        helper.make_node("MatMul", ["se_relu", "se_w2"], ["se_mm2"]),
        helper.make_node("Add", ["se_mm2", "se_b2"], ["se_add2"]),
        helper.make_node("Sigmoid", ["se_add2"], ["se_sig"]),
        helper.make_node("Reshape", ["se_sig", "rs_shape"], ["se_att"]),
        helper.make_node("Mul", ["res1_out", "se_att"], ["se_out"]),
        # LayerNormalization (over last dim = W_res=16)
        helper.make_node("LayerNormalization", ["se_out", "ln_sc", "ln_b"], ["ln_out"],
                         axis=-1, epsilon=1e-5),
        # Dropout (no-op at inference)
        helper.make_node("Dropout", ["ln_out", "drop_ratio"], ["drop_out", "drop_mask"]),
        # 4D MatMul: [1,C,16,16] @ [1,1,16,16] -> [1,C,16,16]
        helper.make_node("MatMul", ["drop_out", "mm4d_w"], ["mm_out"]),
        # Resize nearest asymmetric floor 2x: [1,C,16,16] -> [1,C,32,32]
        helper.make_node("Resize", ["mm_out", "roi_nst", "scales_nst"], ["out"],
                         coordinate_transformation_mode="asymmetric",
                         mode="nearest",
                         nearest_mode="floor"),
    ]

    drop_ratio = np.array(0.0, dtype=np.float32)
    # LayerNorm scale for last dim = W_res=16
    ln_sc_arr = np.ones(W_res, dtype=np.float32)
    ln_b_arr  = np.zeros(W_res, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(empty_roi,   "roi_lin"),
        numpy_helper.from_array(scales_lin,  "scales_lin"),
        numpy_helper.from_array(se_w1,  "se_w1"),
        numpy_helper.from_array(se_b1,  "se_b1"),
        numpy_helper.from_array(se_w2,  "se_w2"),
        numpy_helper.from_array(se_b2,  "se_b2"),
        numpy_helper.from_array(rs_shape, "rs_shape"),
        numpy_helper.from_array(ln_sc_arr, "ln_sc"),
        numpy_helper.from_array(ln_b_arr,  "ln_b"),
        numpy_helper.from_array(drop_ratio, "drop_ratio"),
        numpy_helper.from_array(mm4d_w,  "mm4d_w"),
        numpy_helper.from_array(empty_roi,   "roi_nst"),
        numpy_helper.from_array(scales_nst,  "scales_nst"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0044",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, H, W])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, C, H*4, W*4])],
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
print(f"uid=0044 Resize(linear,asymmetric) + SE-block sigmoid + Resize(nearest,asymmetric)")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"PASS={passed}")
