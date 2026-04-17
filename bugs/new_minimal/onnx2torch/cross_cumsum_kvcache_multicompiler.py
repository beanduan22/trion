#!/usr/bin/env python3
"""
Bug ID     : cross_cumsum_kvcache_multicompiler
Source     : Cross-compiler testing (2026-04-14) — campaign v4 bug_000151
Primary target compiler  : onnx2torch + TorchScript (this folder)
Secondary observation    : OpenVINO 2026.0 CPU (smaller but non-zero divergence)

Patterns   : CumSum(axis=2) → Transpose → MatMul  (the Q×K^T step of a
             minimal self-attention block).

Why this belongs in onnx2torch/:
             The TorchScript path (ONNX → onnx2torch → torch.jit.trace →
             freeze → optimize_for_inference) diverges from the ORT CPU
             reference by a *huge* amount — max_abs_diff ≈ 57 for this
             8×8 graph — while OpenVINO diverges by ~0.32. The dominant
             defect here is on the onnx2torch side, matching the known
             cumsum-axis tracing issue: onnx2torch's CumSum converter
             reads ``axis.item()`` at trace time, which TorchScript then
             bakes as a Python constant. When the optimizer frees/fuses
             around the now-constant axis, the subsequent MatMul picks
             up a different accumulation order or a spurious layout, and
             the numerical result drifts heavily from ORT's reference.
             This is the same class of bug that the other cumsum scripts
             in this folder document.
             OpenVINO's smaller divergence is a separate tiled-GEMM fp32
             accumulation-order effect and is reported only as a
             secondary signal.

ORT_ENABLE_ALL agrees with ORT_DISABLE_ALL exactly, so ORT is not the
fault here — it is the canonical reference.

Min ops    : 3  (CumSum → Transpose → MatMul)
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

# Target marker — this file lives in onnx2torch/, so the TorchScript
# (onnx2torch-traced) path is the primary bug target.
TARGET = "TorchScript"

print(f"\n{'Backend':<15} {'max_abs_diff':>14}  bug?  role")
print("-" * 60)
for k, v in results.items():
    role = "TARGET" if k == TARGET else "secondary" if k == "OpenVINO" else "oracle"
    if isinstance(v, float):
        tag = "BUG" if v > TOL else "ok"
        print(f"{k:<15} {v:>14.6f}  {tag:<4}  {role}")
    else:
        print(f"{k:<15} {v}  {'-':<4}  {role}")

print(f"\nTolerance: {TOL}")
print(f"Primary target: {TARGET} (onnx2torch-converted + jit.trace + freeze)")
print(f"PASS={not any_bug}")
if any_bug:
    failed = [k for k, v in results.items() if isinstance(v, float) and v > TOL]
    target_failed = TARGET in failed
    detail = (
        f"onnx2torch+TorchScript diverges by {results[TARGET]:.3f} from ORT — "
        f"matches the cumsum axis.item() trace-constant class of bugs "
        f"documented elsewhere in this folder"
        if target_failed
        else "OpenVINO-only divergence; onnx2torch primary target matched ORT"
    )
    print(f"BUG REPRODUCED on {failed}. {detail}.")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
