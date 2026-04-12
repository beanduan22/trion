#!/usr/bin/env python3
"""
Bug v2-0042 — ORT: ReduceSum + mul-zero-elim
Compiler : OnnxRuntime
Pattern  : ReduceSum then x*0 eliminated by optimizer before ReduceSum shape is applied
Root cause: ORT optimizer eliminates x*0 (mul-by-zero), but the mul zero-elimination
            can interact with ReduceSum: the broadcast of the constant-zero output
            should carry the post-ReduceSum broadcast shape. ORT_ENABLE_ALL may
            propagate wrong shape information when the zero-mul is folded.

Run: python bug_v2_0042.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(1)

N, C, H, W = 1, 8, 4, 4

def make_model():
    # x [N,C,H,W] -> ReduceSum(axis=2, keepdims=1) -> [N,C,1,W]
    # Add(x, rsm_out): [N,C,H,W] + [N,C,1,W] broadcasts to [N,C,H,W]
    # Mul by zeros -> Add scale (constant = scale)
    nodes = [
        # ReduceSum over H axis with keepdims to allow broadcast
        helper.make_node("ReduceSum", ["x", "rsm_axes"], ["rsm_out"], keepdims=1),
        # Add: [N,C,H,W] + [N,C,1,W] -> [N,C,H,W]
        helper.make_node("Add", ["x", "rsm_out"], ["add_out"]),
        # Mul by zero (ORT optimizer may eliminate this)
        helper.make_node("Mul", ["add_out", "zeros"], ["mul_z"]),
        # Add scale (constant, so after zero-elim result is just scale broadcast)
        helper.make_node("Add", ["mul_z", "scale"], ["out"]),
    ]

    zeros = np.zeros((N, C, H, W), dtype=np.float32)
    scale = np.full((N, C, H, W), 1.05, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(np.array([2], dtype=np.int64), "rsm_axes"),
        numpy_helper.from_array(zeros, "zeros"),
        numpy_helper.from_array(scale, "scale"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0042",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, H, W])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, C, H, W])],
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

# Expected: x*0 + scale = scale (constant), regardless of ReduceSum
expected = np.full((N, C, H, W), 1.05, dtype=np.float32)

max_diff_all = float(np.max(np.abs(out_all - expected)))
max_diff_dis = float(np.max(np.abs(out_dis - expected)))
max_diff_ort = float(np.max(np.abs(out_all - out_dis)))

print(f"uid=0042 ReduceSum + mul-zero-elim")
print(f"  ORT_ENABLE_ALL  vs expected  max_diff={max_diff_all:.6f}")
print(f"  ORT_DISABLE_ALL vs expected  max_diff={max_diff_dis:.6f}")
print(f"  ORT_ENABLE_ALL  vs ORT_DISABLE_ALL max_diff={max_diff_ort:.6f}")
passed = max_diff_ort < 0.01 and max_diff_all < 0.01
print(f"PASS={passed}")
