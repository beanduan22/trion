#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_fp16_matmul_add
Source     : Cross-compiler testing (2026-04-14)
Compiler   : OpenVINO 2026.0, CPU plugin
Patterns   : fp16 MatMul [1,N] × [N,N] + fp16 bias [N]
Root cause : OpenVINO CPU plugin's fp16 tiled GEMM accumulates partial sums
             in a different order than ORT's sequential fp16 path.  fp16 lacks
             the precision to make addition associative at scale, so tile-level
             reordering produces a different result.  Error grows with N:
               N=32  → 0.027   N=64  → 0.078   N=128 → 0.188
             This is the same root cause as ORT#23284 but manifests in OV.
Tolerance  : 0.05

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime as ort
    import openvino as ov
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

np.random.seed(0)
N = 64
W = np.random.randn(N, N).astype(np.float16)
B = np.random.randn(N).astype(np.float16)
x = np.random.randn(1, N).astype(np.float16)

graph = helper.make_graph(
    [helper.make_node("MatMul", ["x", "W"], ["mm"]),
     helper.make_node("Add",    ["mm", "B"], ["y"])],
    "fp16_matmul_add",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, N])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, N])],
    initializer=[numpy_helper.from_array(W, "W"),
                 numpy_helper.from_array(B, "B")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
model_bytes = model.SerializeToString()

# ORT reference (no optimisation)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(model_bytes, so,
      providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

# OpenVINO CPU
core      = ov.Core()
ov_model  = core.read_model(model_bytes, b'')
compiled  = core.compile_model(ov_model, "CPU")
ov_out    = compiled({"x": x})[compiled.output(0)]

diff = float(np.abs(
    ref.astype(np.float64) - ov_out.astype(np.float64)).max())

print(f"ORT ref (first 4): {ref.ravel()[:4]}")
print(f"OpenVINO (first 4): {ov_out.ravel()[:4]}")
print(f"Max abs diff: {diff:.6f}  tol=0.05")

PASS = diff <= 0.05
print(f"PASS={PASS}")
if not PASS:
    print("BUG REPRODUCED: OpenVINO fp16 GEMM tile accumulation diverges from ORT")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
