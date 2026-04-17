#!/usr/bin/env python3
"""
Bug ID     : cross_crms_resize_nearest_ceil
Source     : Cross-compiler testing (2026-04-14) — campaign v4 bug_000078
Compiler   : TorchScript (onnx2torch) fails; ORT and OpenVINO agree
Patterns   : Resize(nearest, half_pixel, nearest_mode=ceil, scale×2)
Root cause : onnx2torch's Resize converter hard-codes PyTorch's floor-based
             nearest interpolation and ignores nearest_mode="ceil".
             For half_pixel coordinates with scale=2, the source pixel for
             destination d is (d+0.5)/2 - 0.5 = d/2 - 0.25.  With ceil
             rounding, d=1 maps to source 1; with floor it maps to source 0.
             Every other destination pixel is assigned the wrong source pixel,
             causing max_diff ≈ 6 on random fp32 inputs.  ORT and OpenVINO
             both implement nearest_mode=ceil correctly.
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

# ── minimal model: a single Resize with nearest_mode=ceil ────────────────────
nodes = [
    helper.make_node("Resize",
                     ["x", "roi", "scales"],
                     ["y"],
                     coordinate_transformation_mode="half_pixel",
                     mode="nearest",
                     nearest_mode="ceil"),
]
inits = [
    numpy_helper.from_array(np.array([], dtype=np.float32), "roi"),
    numpy_helper.from_array(np.array([1., 1., 2., 2.], dtype=np.float32), "scales"),
]
graph = helper.make_graph(nodes, "resize_ceil",
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

# TorchScript via onnx2torch — ignores nearest_mode="ceil", always uses floor
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

print(f"Resize(nearest, half_pixel, nearest_mode=ceil, scale=2)  [{B},{C},{H},{W}]→[{B},{C},{H2},{W2}]")
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
    print(f"BUG REPRODUCED on {bugs}: onnx2torch Resize ignores nearest_mode=ceil, "
          f"uses floor instead — wrong source pixel at every other destination pixel")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
