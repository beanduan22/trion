#!/usr/bin/env python3
"""
Bug v2-0051 — ORT SUSPECT: matmul-scale-softmax + where-mask + softplus
Compiler : OnnxRuntime
Pattern  : MatMul -> scale -> Softmax -> Where-mask -> Mul -> Softplus
Suspect  : Divergence vs pytorch_eager reference; ORT_ENABLE_ALL vs ORT_DISABLE_ALL
           results are close — the bug may be a Softmax axis or Softplus precision difference.
Status   : SUSPECT — shown for completeness; PASS means ORT self-consistent.

Run: python bug_v2_0051.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(10)

N, L, D = 1, 8, 16

def make_model():
    mm_w  = np.random.randn(D, D).astype(np.float32) * 0.1
    ln_sc = np.ones(D, dtype=np.float32)
    ln_b  = np.zeros(D, dtype=np.float32)
    res   = np.zeros((N, L, D), dtype=np.float32)

    nodes = [
        # MatMul + scale
        helper.make_node("MatMul", ["x", "mm_w"], ["mm_out"]),
        helper.make_node("Mul", ["mm_out", "scale"], ["mm_sc"]),
        # Softmax (axis=-1 in original was axis=0, we use -1 for correct semantics)
        helper.make_node("Softmax", ["mm_sc"], ["sm_out"], axis=-1),
        # Where-mask: keep values > threshold, else fill with 0
        helper.make_node("Greater", ["sm_out", "threshold"], ["wh_gt"]),
        helper.make_node("Where", ["wh_gt", "sm_out", "fill_val"], ["wh_out"]),
        # Mul by softmax output (attention-style weighting)
        helper.make_node("Mul", ["wh_out", "sm_out"], ["wmf_out"]),
        # Softplus: log(1 + exp(x))
        helper.make_node("Softplus", ["wmf_out"], ["sp_act"]),
        helper.make_node("Mul", ["sp_act", "sp_one"], ["sp_out"]),
        # Add residual + LayerNorm
        helper.make_node("Add", ["sp_out", "res"], ["aln_add"]),
        helper.make_node("LayerNormalization", ["aln_add", "ln_sc", "ln_b"], ["ln_out"],
                         axis=-1, epsilon=1e-5),
        # Unsqueeze/Squeeze identity (from original)
        helper.make_node("Unsqueeze", ["ln_out", "us_ax"], ["us_out"]),
        helper.make_node("Squeeze",   ["us_out",  "sq_ax"], ["out"]),
    ]

    initializers = [
        numpy_helper.from_array(mm_w, "mm_w"),
        numpy_helper.from_array(np.array([0.125], dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "threshold"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "fill_val"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "sp_one"),
        numpy_helper.from_array(res, "res"),
        numpy_helper.from_array(ln_sc, "ln_sc"),
        numpy_helper.from_array(ln_b,  "ln_b"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "us_ax"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "sq_ax"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0051",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, L, D])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, L, D])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()
x_in = np.random.randn(N, L, D).astype(np.float32)

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_in})[0]

out_all = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

max_diff = float(np.max(np.abs(out_all - out_dis)))
passed = max_diff < 0.01
print(f"uid=0051 SUSPECT matmul-scale-softmax + where-mask + softplus")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"  (Bug was vs pytorch_eager reference, not ORT self-inconsistency)")
print(f"PASS={passed}")
