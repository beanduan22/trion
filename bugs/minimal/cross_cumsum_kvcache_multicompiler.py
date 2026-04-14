#!/usr/bin/env python3
"""
Bug ID     : cross_cumsum_kvcache_multicompiler
Source     : Cross-compiler testing (2026-04-14) — campaign v4 bug_000151
Compiler   : OpenVINO 2026.0 fails (max_diff ≈ 0.16 vs ORT reference)
Patterns   : Add-zero(identity) + CumSum(axis=2) + Mul-Add-Mul chain +
             SE-block (ReduceMean+Mul+Add+Sigmoid+Mul) +
             KV-cache attention (Q×K^T → scale → Softmax → ×V)
Root cause : OpenVINO 2026.0 disagrees with ORT on the combination of:
             (1) CumSum along the feature/channel dimension (axis=2) building
                 up large running sums before the SE-block.
             (2) SE-block: ReduceMean(axis=-1) on the post-CumSum activations
                 interacts with the accumulation pattern, amplifying small
                 fp32 rounding differences between ORT's reference and OV's
                 optimized kernel paths.
             (3) The self-attention (Q×K^T softmax×V) propagates and further
                 amplifies the discrepancy.  ORT_ENABLE_ALL agrees with
                 ORT_DISABLE_ALL exactly; OpenVINO diverges by max_diff≈0.16.
Tolerance  : 0.05

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
try:
    import numpy as np
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime as ort
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

TOL = 0.05
np.random.seed(7)

B, S, D = 1, 16, 32   # batch, seq, dim

zero_addend = np.zeros((B, S, D), dtype=np.float32)
mul_scale   = np.float32(0.5)
add_bias    = np.zeros(D, dtype=np.float32)
se_scale    = np.ones(D, dtype=np.float32)
attn_scale  = np.float32(1.0 / np.sqrt(D))

nodes = [
    # identity add-zero
    helper.make_node("Add", ["x", "zero_addend"], ["id_out"]),
    # CumSum on last axis (simulated causal mask)
    helper.make_node("CumSum", ["id_out", "cumsum_axis"], ["cs_out"],
                     exclusive=0, reverse=0),
    # Mul-Add-Mul chain
    helper.make_node("Mul",  ["cs_out", "mul_scale_c"], ["mul1"]),
    helper.make_node("Add",  ["mul1", "add_bias_c"],    ["add1"]),
    helper.make_node("Mul",  ["add1", "mul_scale_c"],   ["chain_out"]),
    # SE-block: ReduceMean → Mul → Add → Sigmoid → Mul
    helper.make_node("ReduceMean", ["chain_out", "se_axes"], ["se_mean"], keepdims=1),
    helper.make_node("Mul",        ["chain_out", "se_mean"],  ["se_m1"]),
    helper.make_node("Add",        ["se_m1", "se_scale_c"],   ["se_a1"]),
    helper.make_node("Sigmoid",    ["se_a1"],                 ["se_sig"]),
    helper.make_node("Mul",        ["chain_out", "se_sig"],   ["se_out"]),
    # KV-cache attention: Q×K^T → scale → Softmax → × V
    # Transpose K: [B,S,D] → [B,D,S] so Q×K^T = [B,S,D]×[B,D,S] = [B,S,S]
    helper.make_node("Transpose",  ["se_out"],               ["se_out_T"],
                     perm=[0, 2, 1]),
    helper.make_node("MatMul",     ["se_out", "se_out_T"],   ["attn_logits"]),
    helper.make_node("Mul",        ["attn_logits", "attn_sc"], ["attn_sc_out"]),
    helper.make_node("Softmax",    ["attn_sc_out"],           ["attn_w"],
                     axis=-1),
    # V = se_out: [B,S,S]×[B,S,D] = [B,S,D]
    helper.make_node("MatMul",     ["attn_w", "se_out"],     ["y"]),
]

inits = [
    numpy_helper.from_array(zero_addend,                  "zero_addend"),
    numpy_helper.from_array(np.array(2, dtype=np.int64),  "cumsum_axis"),
    numpy_helper.from_array(np.array([-1], dtype=np.int64), "se_axes"),
    numpy_helper.from_array(np.array([[[mul_scale]]]),     "mul_scale_c"),
    numpy_helper.from_array(add_bias,                     "add_bias_c"),
    numpy_helper.from_array(se_scale,                     "se_scale_c"),
    numpy_helper.from_array(np.array([[[attn_scale]]]),   "attn_sc"),
]

graph = helper.make_graph(nodes, "cumsum_kvcache",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, S, D])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, S, D])],  # [1,16,32]
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 8
model_bytes = model.SerializeToString()

x_in = np.random.randn(B, S, D).astype(np.float32)
feed = {"x": x_in}

# ORT reference
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(model_bytes, so_ref,
      providers=["CPUExecutionProvider"]).run(None, feed)[0]

any_bug = False
results = {}

# ORT opt
so_opt = ort.SessionOptions()
so_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opt = ort.InferenceSession(model_bytes, so_opt,
          providers=["CPUExecutionProvider"]).run(None, feed)[0]
d = float(np.abs(ref.astype(np.float64) - ort_opt.astype(np.float64)).max())
results["ORT_opt"] = d
if d > TOL: any_bug = True

# OpenVINO
try:
    import openvino as ov
    core  = ov.Core()
    comp  = core.compile_model(core.read_model(model_bytes, b''), "CPU")
    ov_out = comp(feed)[comp.output(0)]
    d = float(np.abs(ref.astype(np.float64) - ov_out.astype(np.float64)).max())
    results["OpenVINO"] = d
    if d > TOL: any_bug = True
except Exception as e:
    results["OpenVINO"] = f"ERR: {str(e)[:80]}"

# TorchScript
try:
    import torch, onnx, onnx2torch
    net = onnx2torch.convert(onnx.load_from_string(model_bytes))
    net.eval()
    with torch.no_grad():
        ts  = torch.jit.trace(net, torch.from_numpy(x_in))
        jit = ts(torch.from_numpy(x_in)).numpy()
    d = float(np.abs(ref.astype(np.float64) - jit.astype(np.float64)).max())
    results["TorchScript"] = d
    if d > TOL: any_bug = True
except Exception as e:
    results["TorchScript"] = f"ERR: {str(e)[:80]}"

print(f"Input [{B},{S},{D}]")
print(f"ORT_ref (first 4 flat): {ref.ravel()[:4]}")
print()
print(f"{'Backend':<14}  {'max_abs_diff':>14}  {'bug?'}")
print("-" * 36)
for name, val in results.items():
    if isinstance(val, float):
        print(f"{name:<14}  {val:>14.6f}  {'BUG' if val > TOL else 'ok'}")
    else:
        print(f"{name:<14}  {val}")

PASS = not any_bug
print(f"\nTolerance: {TOL}")
print(f"PASS={PASS}")
if not PASS:
    bugs = [k for k, v in results.items() if isinstance(v, float) and v > TOL]
    print(f"BUG REPRODUCED on {bugs}: CumSum+SEblock+KVAttn cross-compiler divergence")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
