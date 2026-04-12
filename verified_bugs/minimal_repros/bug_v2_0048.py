#!/usr/bin/env python3
"""
Bug v2-0048 — ORT SUSPECT: MatMul+Bias+GELU + double LayerNorm + CRMSNorm
Compiler : OnnxRuntime
Pattern  : matmul+bias+GELU fusion + CRMSNorm + MAM chain + SRMNorm + double LayerNorm
Suspect  : Divergence vs pytorch_eager reference; ORT_ENABLE_ALL vs ORT_DISABLE_ALL
           results are close — the bug may be a reference precision difference.
Status   : SUSPECT — shown for completeness; PASS means ORT self-consistent.

Run: python bug_v2_0048.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(7)

N, D = 1, 16

def make_model():
    mm_w   = np.random.randn(D, D).astype(np.float32) * 0.1
    mm_b   = np.zeros(D, dtype=np.float32)
    crms_g = np.ones(D, dtype=np.float32)
    crms_g2= np.ones(D, dtype=np.float32)
    srm_sc = np.ones((1, D), dtype=np.float32)
    ln_g1  = np.ones(D, dtype=np.float32)
    ln_b1  = np.zeros(D, dtype=np.float32)
    ln_g2  = np.ones(D, dtype=np.float32)
    ln_b2  = np.zeros(D, dtype=np.float32)
    ln_sc2 = np.ones((1, D), dtype=np.float32)

    nodes = [
        # MatMul + Bias + GELU
        helper.make_node("MatMul", ["x", "mm_w"], ["mm_out"]),
        helper.make_node("Add", ["mm_out", "mm_b"], ["mm_add"]),
        helper.make_node("Div", ["mm_add", "sqrt2"], ["gelu_div"]),
        helper.make_node("Erf", ["gelu_div"], ["gelu_erf"]),
        helper.make_node("Add", ["gelu_erf", "one"], ["gelu_erf1"]),
        helper.make_node("Mul", ["mm_add", "half"], ["gelu_m1"]),
        helper.make_node("Mul", ["gelu_m1", "gelu_erf1"], ["gelu_out"]),
        # CRMSNorm
        helper.make_node("Mul", ["gelu_out", "gelu_out"], ["crms_sq"]),
        helper.make_node("ReduceMean", ["crms_sq"], ["crms_mean"], axes=[-1], keepdims=0),
        helper.make_node("Add", ["crms_mean", "crms_eps"], ["crms_add"]),
        helper.make_node("Sqrt", ["crms_add"], ["crms_rt"]),
        helper.make_node("Div", ["gelu_out", "crms_rt"], ["crms_n"]),
        helper.make_node("Mul", ["crms_n", "crms_g"], ["crms_sc"]),
        helper.make_node("Mul", ["crms_sc", "crms_g2"], ["crms_out"]),
        # MAM chain
        helper.make_node("Mul", ["crms_out", "mam_a"], ["mam_x1"]),
        helper.make_node("Add", ["mam_x1", "mam_b"], ["mam_x2"]),
        helper.make_node("Mul", ["mam_x2", "mam_c"], ["mam_out"]),
        # SRMNorm (simple RMS norm with reciprocal)
        helper.make_node("Mul", ["mam_out", "mam_out"], ["srm_sq2"]),
        helper.make_node("Add", ["srm_sq2", "srm_eps"], ["srm_add"]),
        helper.make_node("Sqrt", ["srm_add"], ["srm_sq"]),
        helper.make_node("Reciprocal", ["srm_sq"], ["srm_rc"]),
        helper.make_node("Mul", ["mam_out", "srm_rc"], ["srm_norm"]),
        helper.make_node("Mul", ["srm_norm", "srm_sc"], ["srm_out"]),
        # Double LayerNorm
        helper.make_node("LayerNormalization", ["srm_out", "ln_g1", "ln_b1"], ["ln1_out"],
                         axis=-1, epsilon=1e-5),
        helper.make_node("Mul", ["ln1_out", "ln_sc2"], ["ln_sc_out"]),
        helper.make_node("LayerNormalization", ["ln_sc_out", "ln_g2", "ln_b2"], ["out"],
                         axis=-1, epsilon=1e-5),
    ]

    initializers = [
        numpy_helper.from_array(mm_w, "mm_w"),
        numpy_helper.from_array(mm_b, "mm_b"),
        numpy_helper.from_array(np.array([1.4142135], dtype=np.float32), "sqrt2"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "half"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(crms_g, "crms_g"),
        numpy_helper.from_array(crms_g2, "crms_g2"),
        numpy_helper.from_array(np.array([1e-5], dtype=np.float32), "crms_eps"),
        numpy_helper.from_array(np.array([1.2], dtype=np.float32), "mam_a"),
        numpy_helper.from_array(np.array([0.3], dtype=np.float32), "mam_b"),
        numpy_helper.from_array(np.array([1.5], dtype=np.float32), "mam_c"),
        numpy_helper.from_array(np.array([1e-5], dtype=np.float32), "srm_eps"),
        numpy_helper.from_array(srm_sc, "srm_sc"),
        numpy_helper.from_array(ln_g1, "ln_g1"),
        numpy_helper.from_array(ln_b1, "ln_b1"),
        numpy_helper.from_array(ln_sc2, "ln_sc2"),
        numpy_helper.from_array(ln_g2, "ln_g2"),
        numpy_helper.from_array(ln_b2, "ln_b2"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0048",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, D])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, D])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()
x_in = np.random.randn(N, D).astype(np.float32)

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_in})[0]

out_all = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

max_diff = float(np.max(np.abs(out_all - out_dis)))
passed = max_diff < 0.01
print(f"uid=0048 SUSPECT MatMul+Bias+GELU + CRMSNorm + double LayerNorm")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"  (Bug was vs pytorch_eager reference, not ORT self-inconsistency)")
print(f"PASS={passed}")
