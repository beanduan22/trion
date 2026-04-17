#!/usr/bin/env python3
"""
Bug ID     : cross_onnx2torch_resize_nearest_ceil
Source     : Cross-compiler testing (2026-04-13)
Compiler   : onnx2torch / torch.compile
Patterns   : Resize(nearest, ceil, half_pixel, scale 2x)
Root cause : onnx2torch converts ONNX nearest_mode='ceil' to PyTorch F.interpolate which uses floor nearest_mode; torch.compile inherits the wrong behavior. max_diff=3.0 vs ORT reference.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys

try:
    import numpy as np
    import onnx
    from onnx import helper as oh, TensorProto, numpy_helper as onh
    import onnxruntime as ort
    from onnx2torch import convert
    import torch
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

np.random.seed(42)

# Build ONNX model: Resize(nearest, ceil, half_pixel, 2x)
x = np.random.randn(1, 3, 4, 4).astype(np.float32)
node = oh.make_node("Resize", ["X", "roi", "scales"], ["Y"],
    mode="nearest", nearest_mode="ceil",
    coordinate_transformation_mode="half_pixel")
graph = oh.make_graph([node], "test",
    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])],
    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])],
    initializer=[
        onh.from_array(np.array([], dtype=np.float32), "roi"),
        onh.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
model_bytes = model.SerializeToString()

# ORT reference (no optimizations)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(model_bytes, sess_options=so,
    providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

# onnx2torch path
torch_model = convert(model).eval()
with torch.no_grad():
    torch_out = torch_model(torch.from_numpy(x)).numpy()

diff = float(np.max(np.abs(torch_out.astype(np.float64) - ref.astype(np.float64))))
print(f"ORT ref[:4]:      {ref.ravel()[:4]}")
print(f"onnx2torch[:4]:   {torch_out.ravel()[:4]}")
print(f"max_diff={diff:.6f}  tol=0.01")

PASS = diff <= 0.01
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
