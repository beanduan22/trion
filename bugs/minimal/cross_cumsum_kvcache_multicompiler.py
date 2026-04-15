#!/usr/bin/env python3
"""
Bug ID     : cross_cumsum_kvcache_multicompiler
Source     : Cross-compiler testing (2026-04-14) — campaign v4 bug_000151
Compiler   : OpenVINO 2026.0 fails (max_diff ≈ 0.3 vs ORT reference)
Patterns   : CumSum(axis=2) → Transpose → MatMul  (Q×K^T step of self-attention)
Root cause : CumSum along the feature dimension builds up running sums that
             reach magnitude ~D (D=8 here).  The subsequent Q×K^T matmul
             produces dot-products proportional to D^3, large enough that
             OpenVINO's tiled-GEMM kernel accumulates floats in a different
             order than ORT's reference.  Softmax/attention aren't required
             to expose the bug — the matmul output alone diverges by 0.3.
             ORT_ENABLE_ALL agrees with ORT_DISABLE_ALL exactly, so ORT
             does not fuse or reorder these ops.
Min ops    : 3  (CumSum → Transpose → MatMul)  — down from 5
Tolerance  : 0.05

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
from __future__ import annotations

import sys

try:
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:
    print(f"missing deps: {e}")
    sys.exit(2)

TOL: float = 0.05
np.random.seed(7)

B, S, D = 1, 8, 8  # CumSum on axis=2 (D), matmul gives [B,S,S] logits

# ── minimal model: CumSum(axis=2) → Transpose → MatMul ────────────────────
nodes = [
    helper.make_node("CumSum", ["x", "cumsum_axis"], ["cs"]),
    helper.make_node("Transpose", ["cs"], ["cs_T"], perm=[0, 2, 1]),
    helper.make_node("MatMul", ["cs", "cs_T"], ["y"]),
]
inits = [numpy_helper.from_array(np.array(2, dtype=np.int64), "cumsum_axis")]
graph = helper.make_graph(
    nodes,
    "cumsum_matmul_minimal",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, S, D])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, S, S])],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 8
model_bytes = model.SerializeToString()

x_in = np.random.randn(B, S, D).astype(np.float32)
feed = {"x": x_in}

# ORT reference (no optimization)
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(
    model_bytes, so_ref, providers=["CPUExecutionProvider"]
).run(None, feed)[0]

any_bug = False
results: dict[str, float | str] = {}

# ORT_opt should agree with ORT reference (it does for these ops)
so_opt = ort.SessionOptions()
so_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opt = ort.InferenceSession(
    model_bytes, so_opt, providers=["CPUExecutionProvider"]
).run(None, feed)[0]
d = float(np.abs(ref.astype(np.float64) - ort_opt.astype(np.float64)).max())
results["ORT_opt"] = d
if d > TOL:
    any_bug = True

# OpenVINO — expected to diverge
try:
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(core.read_model(model=model_bytes), "CPU")
    ov_out = np.asarray(list(compiled({"x": x_in}).values())[0])
    d = float(np.abs(ref.astype(np.float64) - ov_out.astype(np.float64)).max())
    results["OpenVINO"] = d
    if d > TOL:
        any_bug = True
except Exception as e:
    results["OpenVINO"] = f"unavailable: {str(e)[:80]}"

# TorchScript (frozen + optimize_for_inference)
try:
    import torch
    import onnx2torch

    import onnx

    m = onnx2torch.convert(onnx.load_from_string(model_bytes)).eval()
    t = torch.from_numpy(x_in)
    scripted = torch.jit.trace(m, (t,))
    frozen = torch.jit.optimize_for_inference(torch.jit.freeze(scripted))
    with torch.no_grad():
        out = frozen(t)
    if isinstance(out, (list, tuple)):
        out = out[0]
    ts_out = out.detach().numpy()
    d = float(np.abs(ref.astype(np.float64) - ts_out.astype(np.float64)).max())
    results["TorchScript"] = d
    if d > TOL:
        any_bug = True
except Exception as e:
    results["TorchScript"] = f"unavailable: {str(e)[:80]}"

print(f"\n{'Backend':<15} {'max_abs_diff':>14}  bug?")
print("-" * 40)
for k, v in results.items():
    if isinstance(v, float):
        tag = "BUG" if v > TOL else "ok"
        print(f"{k:<15} {v:>14.6f}  {tag}")
    else:
        print(f"{k:<15} {v}")

print(f"\nTolerance: {TOL}")
print(f"PASS={not any_bug}")
if any_bug:
    failed = [k for k, v in results.items() if isinstance(v, float) and v > TOL]
    print(
        f"BUG REPRODUCED on {failed}: CumSum→Transpose→MatMul "
        f"— OpenVINO tiled-GEMM fp32 accumulation order differs from ORT reference"
    )
    sys.exit(0)
print("not reproduced")
sys.exit(1)
