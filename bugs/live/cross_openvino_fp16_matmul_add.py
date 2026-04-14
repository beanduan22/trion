"""
Bug ID     : cross_openvino_fp16_matmul_add
Source     : Cross-compiler testing (2026-04-14)
Compiler   : OpenVINO 2026.0 (CPU plugin)
Root cause : OpenVINO CPU plugin performs fp16 MatMul accumulation with
             different rounding than ORT (no-opt reference). For a
             [1, 64] × [64, 64] fp16 matrix multiply the intermediate
             partial sums diverge from ORT's row-by-row fp16 accumulation,
             producing max_diff ~0.078 (> 0.05 tolerance).
             The divergence grows with matrix size (size=16 → 0.027,
             size=64 → 0.078, size=128 → 0.188), indicating the OV CPU
             kernel uses a different tile-based fp16 accumulation order.
Tolerance  : 0.05
Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys
import numpy as np

try:
    import onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
    import openvino as ov
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(2)

np.random.seed(42)
size = 64
W = np.random.randn(size, size).astype(np.float16)
B = np.random.randn(size).astype(np.float16)
x = np.random.randn(1, size).astype(np.float16)

# Build ONNX fp16 MatMul + Add graph
W_init = onh.from_array(W, "W")
B_init = onh.from_array(B, "B")
nodes = [
    oh.make_node("MatMul", ["x", "W"], ["mm_out"]),
    oh.make_node("Add",    ["mm_out", "B"], ["Y"]),
]
g = oh.make_graph(nodes, "t",
    [oh.make_tensor_value_info("x", TP.FLOAT16, [1, size])],
    [oh.make_tensor_value_info("Y", TP.FLOAT16, [1, size])],
    initializer=[W_init, B_init])
m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17)])
m.ir_version = 8
mb = m.SerializeToString()

# ORT no-opt reference
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so,
                           providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

# OpenVINO CPU
core = ov.Core()
mv = core.read_model(mb, b"")
c = core.compile_model(mv, "CPU")
ov_out = c({"x": x})[c.output(0)]

max_diff = float(np.max(np.abs(
    ov_out.astype(np.float64) - ref.astype(np.float64)
)))

print(f"ORT ref (first 4): {ref[0, :4]}")
print(f"OpenVINO  (first 4): {ov_out[0, :4]}")
print(f"Max abs diff: {max_diff:.6f}")
print(f"BUG REPRODUCED: {max_diff > 0.05}")

# Exit 0 if bug reproduced (diff > tolerance), 1 if fixed
sys.exit(0 if max_diff > 0.05 else 1)
