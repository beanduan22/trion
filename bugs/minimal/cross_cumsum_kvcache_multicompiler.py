#!/usr/bin/env python3
"""
Bug ID     : cross_cumsum_kvcache_multicompiler
Source     : Cross-compiler testing (2026-04-14) — campaign v4 bug_000151
Compiler   : OpenVINO 2026.0 fails (max_diff ≈ 0.07–3.2 vs ORT reference)
Patterns   : CumSum(axis=2) + self-attention (Q×K^T → Softmax → ×V)
Root cause : CumSum along the feature dimension builds up running sums that
             serve as Q, K, and V in self-attention.  The resulting Q×K^T
             values are numerically large (proportional to D²).  OpenVINO's
             tiled-GEMM kernel accumulates the matrix product with a different
             floating-point summation order than ORT's reference implementation,
             causing divergence that is then amplified by Softmax's exponential
             sensitivity.  ORT_ENABLE_ALL agrees with ORT_DISABLE_ALL exactly;
             ORT does not fuse or reorder these ops.
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

B, S, D = 1, 8, 8   # CumSum on D=8 axis, self-attention over S=8 tokens

# ── minimal model: CumSum(axis=2) → self-attention ──────────────────────────
nodes = [
    helper.make_node("CumSum", ["x", "cumsum_axis"], ["cs"],
                     exclusive=0, reverse=0),
    # Q=cs, K^T=cs^T: [B,S,D]×[B,D,S] = [B,S,S]
    helper.make_node("Transpose", ["cs"], ["cs_T"], perm=[0, 2, 1]),
    helper.make_node("MatMul",    ["cs", "cs_T"],   ["logits"]),
    helper.make_node("Softmax",   ["logits"],        ["attn_w"], axis=-1),
    # attn_w×V: [B,S,S]×[B,S,D] = [B,S,D]
    helper.make_node("MatMul",    ["attn_w", "cs"], ["y"]),
]

inits = [
    numpy_helper.from_array(np.array(2, dtype=np.int64), "cumsum_axis"),
]

graph = helper.make_graph(nodes, "cumsum_self_attn",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, S, D])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, S, D])],
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 8
model_bytes = model.SerializeToString()

x_in = np.random.randn(B, S, D).astype(np.float32)
feed = {"x": x_in}

# ORT reference (no optimization)
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(model_bytes, so_ref,
      providers=["CPUExecutionProvider"]).run(None, feed)[0]

any_bug = False
results = {}

# ORT_opt
so_opt = ort.SessionOptions()
so_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opt = ort.InferenceSession(model_bytes, so_opt,
          providers=["CPUExecutionProvider"]).run(None, feed)[0]
d = float(np.abs(ref.astype(np.float64) - ort_opt.astype(np.float64)).max())
results["ORT_opt"] = d
if d > TOL:
    any_bug = True

# OpenVINO
try:
    import openvino as ov
    core  = ov.Core()
    comp  = core.compile_model(core.read_model(model_bytes, b''), "CPU")
    ov_out = comp(feed)[comp.output(0)]
    d = float(np.abs(ref.astype(np.float64) - ov_out.astype(np.float64)).max())
    results["OpenVINO"] = d
    if d > TOL:
        any_bug = True
except Exception as e:
    results["OpenVINO"] = f"ERR: {str(e)[:80]}"

# TorchScript via onnx2torch
try:
    import onnx, onnx2torch, torch
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = onnx2torch.convert(onnx.load_from_string(model_bytes))
        net.eval()
        ts  = torch.jit.trace(net, torch.from_numpy(x_in))
        jit_out = ts(torch.from_numpy(x_in)).detach().numpy()
    d = float(np.abs(ref.astype(np.float64) - jit_out.astype(np.float64)).max())
    results["TorchScript"] = d
    if d > TOL:
        any_bug = True
except Exception as e:
    results["TorchScript"] = f"ERR: {str(e)[:80]}"

print(f"CumSum(axis=2) + self-attention  input [{B},{S},{D}]")
print(f"ORT_ref (first 4 flat): {ref.ravel()[:4]}")
print()
print(f"{'Backend':<14}  {'max_abs_diff':>14}  {'bug?'}")
print("-" * 36)
for name, val in results.items():
    if isinstance(val, float):
        print(f"{name:<14}  {val:>14.6f}  {'BUG' if val > TOL else 'ok'}")
    else:
        print(f"{name:<14}  {val}")

print(f"\nTolerance: {TOL}")
PASS = not any_bug
print(f"PASS={PASS}")
if not PASS:
    bugs = [k for k, v in results.items() if isinstance(v, float) and v > TOL]
    print(f"BUG REPRODUCED on {bugs}: CumSum(axis=2)+self-attention — "
          f"OpenVINO tiled-GEMM fp32 accumulation order differs from ORT reference")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
