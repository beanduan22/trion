#!/usr/bin/env python3
"""
Bug v2-0040 — ORT SUSPECT: gather_reshape + ReduceL2 + MatMul+Bias+GELU + LayerNorm
Compiler : OnnxRuntime
Pattern  : gather_reshape + reduce_l2_last + matmul+bias+GELU + layernorm
Suspect  : Divergence vs pytorch_eager reference; ORT_ENABLE_ALL vs ORT_DISABLE_ALL
           results are close — the bug may be a reference precision difference.
Status   : SUSPECT — shown for completeness; PASS means ORT self-consistent.

Run: python bug_v2_0040.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(42)

# Weights
emb_w = np.random.randn(64, 32).astype(np.float32) * 0.1
mm_w  = np.random.randn(32, 64).astype(np.float32) * 0.1
mm_b  = np.random.randn(64).astype(np.float32) * 0.1
ln_sc = np.ones(64, dtype=np.float32)
ln_b  = np.zeros(64, dtype=np.float32)

def make_model():
    # Graph: Gather(emb, idx) -> Reshape -> ReduceL2(last) -> Mul(zero)+Add(norm) ->
    #        ManualLayerNorm -> MatMul+Bias+GELU -> LayerNorm -> Dropout
    nodes = [
        helper.make_node("Gather", ["emb", "idx"], ["gathered"], axis=0),
        helper.make_node("Reshape", ["gathered", "oshape"], ["reshaped"]),
        # ReduceL2 on last axis
        helper.make_node("ReduceL2", ["reshaped"], ["l2_red"], axes=[-1], keepdims=0),
        # mul by zero then add l2 (keeps l2 value, zero-elim pattern)
        helper.make_node("Mul", ["reshaped", "zero_scale"], ["mul_z"]),
        helper.make_node("Add", ["mul_z", "l2_red"], ["after_l2"]),
        # Manual LayerNorm
        helper.make_node("ReduceMean", ["after_l2"], ["mln_m"], axes=[-1], keepdims=0),
        helper.make_node("Sub", ["after_l2", "mln_m"], ["mln_sub"]),
        helper.make_node("Mul", ["mln_sub", "mln_sub"], ["mln_sq"]),
        helper.make_node("ReduceMean", ["mln_sq"], ["mln_var"], axes=[-1], keepdims=0),
        helper.make_node("Add", ["mln_var", "mln_eps"], ["mln_vadd"]),
        helper.make_node("Sqrt", ["mln_vadd"], ["mln_sqrt"]),
        helper.make_node("Div", ["mln_sub", "mln_sqrt"], ["mln_out"]),
        # MatMul + Bias + GELU
        helper.make_node("MatMul", ["mln_out", "mm_w"], ["mm_out"]),
        helper.make_node("Add", ["mm_out", "mm_b"], ["mm_add"]),
        helper.make_node("Div", ["mm_add", "sqrt2"], ["gelu_div"]),
        helper.make_node("Erf", ["gelu_div"], ["gelu_erf"]),
        helper.make_node("Add", ["gelu_erf", "one"], ["gelu_erf1"]),
        helper.make_node("Mul", ["mm_add", "half"], ["gelu_m1"]),
        helper.make_node("Mul", ["gelu_m1", "gelu_erf1"], ["gelu_out"]),
        # LayerNormalization (ORT may fuse)
        helper.make_node("LayerNormalization", ["gelu_out", "ln_sc", "ln_b"], ["ln_out"],
                         axis=-1, epsilon=1e-5),
    ]

    initializers = [
        numpy_helper.from_array(emb_w, "emb"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "idx"),
        numpy_helper.from_array(np.array([1, 32], dtype=np.int64), "oshape"),
        numpy_helper.from_array(np.zeros((1,), dtype=np.float32), "zero_scale"),
        numpy_helper.from_array(np.array([1e-5], dtype=np.float32), "mln_eps"),
        numpy_helper.from_array(mm_w, "mm_w"),
        numpy_helper.from_array(mm_b, "mm_b"),
        numpy_helper.from_array(np.array([1.4142135], dtype=np.float32), "sqrt2"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "half"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(ln_sc, "ln_sc"),
        numpy_helper.from_array(ln_b, "ln_b"),
    ]

    graph = helper.make_graph(
        nodes,
        "bug_v2_0040",
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, [64, 32])],
        [helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, [1, 64])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()

# Input: embedding table is an initializer; the "model_input" to vary is the embedding weights
# For this suspect bug, we use fixed input and compare ORT_ENABLE_ALL vs ORT_DISABLE_ALL
x = np.random.randn(64, 32).astype(np.float32) * 0.1

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    # Input is the embedding table
    out = sess.run(None, {"emb": x})[0]
    return out

out_all = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

max_diff = float(np.max(np.abs(out_all - out_dis)))
# SUSPECT: expect self-consistency (diff near 0)
passed = max_diff < 0.01
print(f"uid=0040 SUSPECT gather_reshape+ReduceL2+GELU+LayerNorm")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"  (Bug was vs pytorch_eager reference, not ORT self-inconsistency)")
print(f"PASS={passed}")
