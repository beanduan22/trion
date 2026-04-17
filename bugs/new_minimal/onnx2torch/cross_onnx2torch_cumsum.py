#!/usr/bin/env python3
"""
Bug ID     : cross_onnx2torch_cumsum
Source     : Cross-compiler testing (2026-04-13)
Compiler   : onnx2torch / torch.compile
Patterns   : CumSum(axis=2) on 3D tensor [1,4,8]
Root cause : onnx2torch CumSum converter calls axis.item() which causes torch.compile graph break; the broken graph produces wrong cumulative sum with max_diff=8.63 vs ORT reference.
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

# Build ONNX model: CumSum along last axis
x = np.random.randn(1, 4, 8).astype(np.float32)
node = oh.make_node("CumSum", ["X", "axis"], ["Y"])
graph = oh.make_graph([node], "test",
    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8])],
    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8])],
    initializer=[onh.from_array(np.array([2], dtype=np.int64), "axis")])
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
print(f"ORT ref[:8]:      {ref.ravel()[:8]}")
print(f"onnx2torch[:8]:   {torch_out.ravel()[:8]}")
print(f"max_diff={diff:.6f}  tol=0.01")

PASS = diff <= 0.01
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
