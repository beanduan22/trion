#!/usr/bin/env python3
"""
Bug v2-0049 — ORT: MatMul+Bias+GELU fusion fires when gated output is zeroed
Compiler : OnnxRuntime
Pattern  : MatMul+Bias+GELU -> Gated Linear Block -> zero-mask -> softmax
Root cause: ORT fuses the MatMul+Bias+GELU pattern into a single GeLU kernel.
            When the output gate zeros out gelu_out (via x*zeros + scale), the
            fused GELU kernel still operates on the un-zeroed value internally,
            producing output that should be `scale` (constant) but ORT_ENABLE_ALL
            may carry wrong values through the fused path.

Run: python bug_v2_0049.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(8)

N, D_in, D_out = 1, 16, 32

def make_model():
    mm_w  = np.random.randn(D_in, D_out).astype(np.float32) * 0.1
    mm_b  = np.zeros(D_out, dtype=np.float32)
    glb_w1 = np.random.randn(D_out, D_out).astype(np.float32) * 0.1
    glb_w2 = np.random.randn(D_out, D_out).astype(np.float32) * 0.1
    glb_b  = np.zeros(D_out, dtype=np.float32)
    zeros  = np.zeros((1, D_out), dtype=np.float32)
    scale  = np.full((1, D_out), 1.05, dtype=np.float32)
    ln_sc  = np.ones(D_out, dtype=np.float32)
    ln_b   = np.zeros(D_out, dtype=np.float32)
    res    = np.zeros((1, D_out), dtype=np.float32)

    nodes = [
        # MatMul + Bias + GELU
        helper.make_node("MatMul", ["x", "mm_w"], ["mm_out"]),
        helper.make_node("Add", ["mm_out", "mm_b"], ["mm_add"]),
        helper.make_node("Div", ["mm_add", "sqrt2"], ["gelu_div"]),
        helper.make_node("Erf", ["gelu_div"], ["gelu_erf"]),
        helper.make_node("Add", ["gelu_erf", "one"], ["gelu_erf1"]),
        helper.make_node("Mul", ["mm_add", "half"], ["gelu_m1"]),
        helper.make_node("Mul", ["gelu_m1", "gelu_erf1"], ["gelu_out"]),
        # Gated Linear Block
        helper.make_node("Gemm", ["gelu_out", "glb_w1", "glb_b"], ["glb_g1"],
                         alpha=1.0, beta=1.0, transB=0),
        helper.make_node("Gemm", ["gelu_out", "glb_w2", "glb_b"], ["glb_g2"],
                         alpha=1.0, beta=1.0, transB=0),
        helper.make_node("Sigmoid", ["glb_g2"], ["glb_sig"]),
        helper.make_node("Mul", ["glb_g1", "glb_sig"], ["glb_out"]),
        # Zero-mask: output * zeros + scale = constant (scale)
        # ORT_ENABLE_ALL may fuse GELU and then fail to constant-fold this correctly
        helper.make_node("Mul", ["glb_out", "zeros"], ["mze_z"]),
        helper.make_node("Add", ["mze_z", "scale"], ["mze_out"]),
        # Stable softmax
        helper.make_node("ReduceMax", ["mze_out"], ["sm_max"], axes=[-1], keepdims=0),
        helper.make_node("Sub", ["mze_out", "sm_max"], ["sm_sub"]),
        helper.make_node("Exp", ["sm_sub"], ["sm_exp"]),
        helper.make_node("ReduceSum", ["sm_exp", "sm_ax"], ["sm_sum"], keepdims=0),
        helper.make_node("Div", ["sm_exp", "sm_sum"], ["sm_dv"]),
        helper.make_node("Mul", ["sm_dv", "mze_out"], ["sm_out"]),
        # LayerNorm + residual
        helper.make_node("LayerNormalization", ["sm_out", "ln_sc", "ln_b"], ["ln_out"],
                         axis=-1, epsilon=1e-5),
        helper.make_node("Add", ["ln_out", "res"], ["out"]),
    ]

    sm_ax = np.array([-1], dtype=np.int64)

    initializers = [
        numpy_helper.from_array(mm_w, "mm_w"),
        numpy_helper.from_array(mm_b, "mm_b"),
        numpy_helper.from_array(np.array([1.4142135], dtype=np.float32), "sqrt2"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "half"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(glb_w1, "glb_w1"),
        numpy_helper.from_array(glb_w2, "glb_w2"),
        numpy_helper.from_array(glb_b, "glb_b"),
        numpy_helper.from_array(zeros, "zeros"),
        numpy_helper.from_array(scale, "scale"),
        numpy_helper.from_array(sm_ax, "sm_ax"),
        numpy_helper.from_array(ln_sc, "ln_sc"),
        numpy_helper.from_array(ln_b,  "ln_b"),
        numpy_helper.from_array(res,   "res"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0049",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, D_in])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, D_out])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()

# Use varied inputs to expose the bug
x_in = np.random.randn(N, D_in).astype(np.float32) * 2.0

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_in})[0]

out_all = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

max_diff = float(np.max(np.abs(out_all - out_dis)))

# After zero-mask: mze_out = scale = 1.05 everywhere
# softmax of constant = uniform = 1/D_out
# sm_out = uniform * 1.05 = 1.05/D_out
sm_uniform = np.full((N, D_out), 1.05 / D_out, dtype=np.float32)
# LayerNorm of constant = 0 (mean=val, var=0) -> 0*scale + bias = bias
# With scale=1, bias=0: output = 0 + res = res = 0
expected = np.zeros((N, D_out), dtype=np.float32)

max_diff_exp_all = float(np.max(np.abs(out_all - expected)))
max_diff_exp_dis = float(np.max(np.abs(out_dis - expected)))

print(f"uid=0049 MatMul+Bias+GELU fusion with gated zero-mask output")
print(f"  After zero-mask output must be constant=1.05; softmax+layernorm -> 0")
print(f"  ORT_ENABLE_ALL  vs expected(0) max_diff={max_diff_exp_all:.6f}")
print(f"  ORT_DISABLE_ALL vs expected(0) max_diff={max_diff_exp_dis:.6f}")
print(f"  ORT_ENABLE_ALL  vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
passed = max_diff < 0.01
print(f"PASS={passed}")
