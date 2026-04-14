#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_conv_fp32_precision
Source     : Cross-compiler testing (2026-04-14)
Compiler   : OpenVINO 2026.0, CPU plugin
Patterns   : Conv [1,C,H,W] with random float32 weights, no BN or bias
Root cause : OpenVINO's CPU plugin selects a Winograd / tiled GEMM algorithm
             for float32 Conv that accumulates dot products in a different order
             than ORT's direct-convolution reference path.  For a 4×4×3×3 weight
             matrix the partial sums (36 multiplications per output element) can
             diverge by up to 0.054 due to floating-point reassociation.  This
             is distinct from the Conv+BN fusion bug (cross_openvino_conv_bn_fusion)
             which also has a systematic rounding component.  Error increases with
             larger kernel size and channel count.
Tolerance  : 0.02

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

TOL = 0.02

np.random.seed(0)
B, C, H, W = 1, 4, 8, 8
kH, kW     = 3, 3
W_conv     = np.random.randn(C, C, kH, kW).astype(np.float32)
x_in       = np.random.randn(B, C, H, W).astype(np.float32)

nodes = [helper.make_node("Conv", ["x", "W_conv"], ["y"], pads=[1, 1, 1, 1])]
inits = [numpy_helper.from_array(W_conv, "W_conv")]
graph = helper.make_graph(nodes, "conv_fp32_prec",
    [helper.make_tensor_value_info("x",     TensorProto.FLOAT, [B, C, H, W])],
    [helper.make_tensor_value_info("y",     TensorProto.FLOAT, [B, C, H, W])],
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
model_bytes = model.SerializeToString()

# ORT reference (no optimisation)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(model_bytes, so,
      providers=["CPUExecutionProvider"]).run(None, {"x": x_in})[0]

# OpenVINO CPU
core     = ov.Core()
ov_model = core.read_model(model_bytes, b'')
compiled = core.compile_model(ov_model, "CPU")
ov_out   = compiled({"x": x_in})[compiled.output(0)]

diff = float(np.abs(ref.astype(np.float64) - ov_out.astype(np.float64)).max())

print(f"ORT ref (first 4): {ref.ravel()[:4]}")
print(f"OpenVINO (first 4): {ov_out.ravel()[:4]}")
print(f"Max abs diff: {diff:.6f}  tol={TOL}")

# Also test a larger kernel (5x5) — more accumulations → more error
kH2, kW2 = 5, 5
W2 = np.random.randn(C, C, kH2, kW2).astype(np.float32)
nodes2 = [helper.make_node("Conv", ["x", "W2"], ["y"], pads=[2, 2, 2, 2])]
inits2 = [numpy_helper.from_array(W2, "W2")]
graph2 = helper.make_graph(nodes2, "conv5x5",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, C, H, W])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, C, H, W])],
    initializer=inits2)
model2 = helper.make_model(graph2, opset_imports=[helper.make_opsetid("", 13)])
model2.ir_version = 8
mb2 = model2.SerializeToString()

ref2  = ort.InferenceSession(mb2, so, providers=["CPUExecutionProvider"]).run(None, {"x": x_in})[0]
ov2   = core.compile_model(core.read_model(mb2, b''), "CPU")({"x": x_in})[0]
diff2 = float(np.abs(ref2.astype(np.float64) - ov2.astype(np.float64)).max())
print(f"5×5 kernel max abs diff: {diff2:.6f}  tol={TOL}")

PASS = diff <= TOL and diff2 <= TOL
print(f"PASS={PASS}")
if not PASS:
    print("BUG REPRODUCED: OpenVINO CPU Conv fp32 accumulation differs from ORT reference")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
