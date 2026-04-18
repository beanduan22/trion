#!/usr/bin/env python3
"""
Bug ID     : cross_bitshift_shift64_ov_ort
Source     : Cross-compiler testing (2026-04-14) — extends github_inductor_011
Compiler   : OnnxRuntime 1.24.4 AND OpenVINO 2026.0 — both fail
Patterns   : ONNX BitShift (direction=RIGHT) with shift amount == 64 on uint64
Root cause : Both ORT and OpenVINO generate code that calls the underlying
             C/hardware right-shift: `x >> 64`.  The C standard defines this as
             undefined behaviour; on x86 the SHRQ/SHR instruction masks the
             shift count to the low 6 bits (shift & 63), so `x >> 64` becomes
             `x >> 0 = x` instead of 0.  OpenVINO's CPU plugin is also affected
             (same code-generation path), confirmed: [1000, 255] >> [64, 64]
             returns [1000, 255] instead of [0, 0].  ORT_ENABLE_ALL happens to
             constant-fold the expression to 0 in this specific test case.
Tolerance  : exact equality (result MUST be 0 for shift == 64 on any uint64 value)

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

# ── model: BitShift RIGHT with shift=64 on uint64 ────────────────────────────
vals   = np.array([[1000, 255, 1, 42]], dtype=np.uint64)
shifts = np.array([[64,   64, 64, 64]], dtype=np.uint64)
expected = np.zeros_like(vals)

nodes = [helper.make_node("BitShift", ["x", "n"], ["y"], direction="RIGHT")]
graph = helper.make_graph(nodes, "bitshift64",
    [helper.make_tensor_value_info("x", TensorProto.UINT64, list(vals.shape)),
     helper.make_tensor_value_info("n", TensorProto.UINT64, list(shifts.shape))],
    [helper.make_tensor_value_info("y", TensorProto.UINT64, list(vals.shape))])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8
model_bytes = model.SerializeToString()

feed = {"x": vals, "n": shifts}

any_bug = False
results = {}

# ORT no-optimization (reference: should compute 0)
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref_out = ort.InferenceSession(model_bytes, so_ref,
          providers=["CPUExecutionProvider"]).run(None, feed)[0]
ref_diff = int(np.abs(ref_out.astype(np.int64) - expected.astype(np.int64)).max())
results["ORT_noopt"] = ref_diff
if ref_diff > 0:
    any_bug = True

# ORT all-optimization
so_opt = ort.SessionOptions()
so_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opt_out = ort.InferenceSession(model_bytes, so_opt,
          providers=["CPUExecutionProvider"]).run(None, feed)[0]
opt_diff = int(np.abs(opt_out.astype(np.int64) - expected.astype(np.int64)).max())
results["ORT_opt"] = opt_diff
if opt_diff > 0:
    any_bug = True

# OpenVINO
try:
    import openvino as ov
    core  = ov.Core()
    comp  = core.compile_model(core.read_model(model_bytes, b''), "CPU")
    ov_out = comp(feed)[comp.output(0)]
    ov_diff = int(np.abs(ov_out.astype(np.int64) - expected.astype(np.int64)).max())
    results["OpenVINO"] = ov_diff
    if ov_diff > 0:
        any_bug = True
except Exception as e:
    results["OpenVINO"] = f"ERR: {str(e)[:80]}"

# Left shift by 64 also (same UB)
nodes_l = [helper.make_node("BitShift", ["x", "n"], ["y"], direction="LEFT")]
graph_l  = helper.make_graph(nodes_l, "bitshift64_left",
    [helper.make_tensor_value_info("x", TensorProto.UINT64, list(vals.shape)),
     helper.make_tensor_value_info("n", TensorProto.UINT64, list(shifts.shape))],
    [helper.make_tensor_value_info("y", TensorProto.UINT64, list(vals.shape))])
model_l   = helper.make_model(graph_l, opset_imports=[helper.make_opsetid("", 11)])
model_l.ir_version = 8
mb_left   = model_l.SerializeToString()

left_ref  = ort.InferenceSession(mb_left, so_ref,
            providers=["CPUExecutionProvider"]).run(None, feed)[0]
left_diff = int(np.abs(left_ref.astype(np.int64) - expected.astype(np.int64)).max())
results["ORT_noopt_LEFT"] = left_diff
if left_diff > 0:
    any_bug = True

try:
    import openvino as ov
    core  = ov.Core()
    comp_l = core.compile_model(core.read_model(mb_left, b''), "CPU")
    ov_left = comp_l(feed)[comp_l.output(0)]
    left_ov = int(np.abs(ov_left.astype(np.int64) - expected.astype(np.int64)).max())
    results["OpenVINO_LEFT"] = left_ov
    if left_ov > 0:
        any_bug = True
except Exception as e:
    results["OpenVINO_LEFT"] = f"ERR: {str(e)[:80]}"

print(f"Input values: {vals[0].tolist()}")
print(f"Shift amount: {shifts[0].tolist()} (all 64)")
print(f"Expected:     {expected[0].tolist()} (all 0)")
print()
print(f"{'Backend':<20}  {'max_diff':>10}  {'bug?'}")
print("-" * 40)
for name, val in results.items():
    direction = "(LEFT)" if "LEFT" in name else "(RIGHT)"
    if isinstance(val, int):
        print(f"{name:<20}  {val:>10}  {'BUG' if val > 0 else 'ok'}")
    else:
        print(f"{name:<20}  {val}")

PASS = not any_bug
print(f"\nPASS={PASS}")
if not PASS:
    bugs = [k for k, v in results.items() if isinstance(v, int) and v > 0]
    print(f"BUG REPRODUCED on {bugs}: BitShift by 64 returns non-zero (C UB — x86 masks shift to 6 bits)")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
