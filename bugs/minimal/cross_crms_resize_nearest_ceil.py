#!/usr/bin/env python3
"""
Bug ID     : cross_crms_resize_nearest_ceil
Source     : Cross-compiler testing (2026-04-14) — campaign v4 bug_000078
Compiler   : TorchScript (onnx2torch) fails; ORT and OpenVINO agree
Patterns   : cRMS-norm (Mul→ReduceMean→Add→Sqrt→Div→Mul×2) +
             Resize(nearest, half_pixel, nearest_mode=ceil, scale×2)
Root cause : onnx2torch's Resize converter ignores nearest_mode="ceil" and
             always uses PyTorch's floor-based nearest interpolation.
             For half_pixel coordinates, ceil vs floor choose different
             source pixels at every other destination pixel, producing
             large diffs (max_diff ≈ 5.6 for random fp32 inputs).
             ORT and OpenVINO both implement ceil correctly.
Tolerance  : 0.05

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime as ort
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

TOL = 0.05
np.random.seed(42)

B, C, H, W = 1, 3, 32, 32
H2, W2 = H * 2, W * 2

# cRMS-norm learned scales (non-trivial to make the output interesting)
crms_g1 = np.ones(C * H * W // (C * H * W // C), dtype=np.float32)  # identity scale
crms_g1 = np.ones(C, dtype=np.float32)
# Per-channel second scale (from v4_078 initializer, rounded)
crms_g2 = np.array([1.013, 0.987, 1.064], dtype=np.float32)
crms_eps = np.array([1e-6], dtype=np.float32)

# ── build model ──────────────────────────────────────────────────────────────
# cRMS-norm reduces over last dim (width), then Resize×2 with nearest+ceil
nodes = [
    helper.make_node("Mul",        ["x", "x"],             ["sq"]),
    # opset 17: axes as attribute (int list)
    helper.make_node("ReduceMean", ["sq"],                  ["ms"],
                     axes=[-1], keepdims=1),
    helper.make_node("Add",        ["ms", "crms_eps"],      ["add"]),
    helper.make_node("Sqrt",       ["add"],                 ["rt"]),
    helper.make_node("Div",        ["x", "rt"],             ["norm"]),
    helper.make_node("Mul",        ["norm", "crms_g1"],     ["sc"]),
    helper.make_node("Mul",        ["sc", "crms_g2"],       ["out"]),
    # Resize: half_pixel + nearest_mode=ceil is the tricky combination
    helper.make_node("Resize",
                     ["out", "roi", "scales"],
                     ["y"],
                     coordinate_transformation_mode="half_pixel",
                     mode="nearest",
                     nearest_mode="ceil"),
]

inits = [
    numpy_helper.from_array(crms_g1.reshape(C, 1, 1),            "crms_g1"),
    numpy_helper.from_array(crms_g2.reshape(C, 1, 1),            "crms_g2"),
    numpy_helper.from_array(crms_eps,                             "crms_eps"),
    numpy_helper.from_array(np.array([], dtype=np.float32),       "roi"),
    numpy_helper.from_array(np.array([1., 1., 2., 2.],
                                     dtype=np.float32),           "scales"),
]

graph = helper.make_graph(nodes, "crms_resize_ceil",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, C, H, W])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, C, H2, W2])],
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
model_bytes = model.SerializeToString()

x_in = np.random.randn(B, C, H, W).astype(np.float32)
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
try:
    ort_opt = ort.InferenceSession(model_bytes, so_opt,
               providers=["CPUExecutionProvider"]).run(None, feed)[0]
    d = float(np.abs(ref.astype(np.float64) - ort_opt.astype(np.float64)).max())
    results["ORT_opt"] = d
    if d > TOL:
        any_bug = True
except Exception as e:
    results["ORT_opt"] = f"ERR: {e}"

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

# TorchScript via onnx2torch — ignores nearest_mode="ceil", uses floor instead
try:
    import torch, onnx2torch
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

print(f"Input shape: [{B},{C},{H},{W}]  →  Resize(×2) nearest+ceil  →  [{B},{C},{H2},{W2}]")
print(f"ORT_ref (first 4 flat): {ref.ravel()[:4]}")
print()
print(f"{'Compiler':<14}  {'max_abs_diff':>14}  {'bug?'}")
print("-" * 38)
for comp_name, val in results.items():
    if isinstance(val, float):
        print(f"{comp_name:<14}  {val:>14.6f}  {'BUG' if val > TOL else 'ok'}")
    else:
        print(f"{comp_name:<14}  {val}")

print(f"\nTolerance: {TOL}")
PASS = not any_bug
print(f"PASS={PASS}")
if not PASS:
    bugs = [k for k, v in results.items() if isinstance(v, float) and v > TOL]
    print(f"BUG REPRODUCED on {bugs}: Resize(nearest_mode=ceil) — "
          f"onnx2torch uses floor instead of ceil, causing large pixel-selection errors")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
