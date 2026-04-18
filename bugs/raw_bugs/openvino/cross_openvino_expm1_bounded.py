#!/usr/bin/env python3
"""
Bug: OpenVINO CPU fp32 Exp→MatMul accumulation differs from ORT (expm1_bounded pattern)
Compiler: OpenVINO
Oracle:   ORT_DISABLE_ALL

Root cause: OV CPU GEMM tiles the inner-dimension sum in a different order than
ORT's sequential accumulation. The pattern Exp(x) → MatMul(W) with a wide inner
dimension (K=256) amplifies the accumulation difference. Inputs are bounded so
that Exp(x) stays finite but the resulting reciprocal sum is sensitive to order.

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys
try:
    import numpy as np, onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort, openvino as ov
except ImportError as e:
    print(f"missing dep: {e}"); sys.exit(2)

np.random.seed(42)
x = np.random.randn(2, 256).astype(np.float32) * 0.5   # bounded → finite Exp values
W = np.random.randn(256, 64).astype(np.float32)

nodes = [
    oh.make_node("Exp",    ["X"],    ["e"]),
    oh.make_node("MatMul", ["e","W"],["Y"]),
]
graph = oh.make_graph(nodes, "expm1_bounded",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2, 256])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,  64])],
    initializer=[onh.from_array(W, "W")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref_out = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

core = ov.Core()
comp = core.compile_model(core.read_model(mb, b""), "CPU")
ov_out = np.array(comp({"X": x})[comp.output(0)])

max_abs = float(np.abs(ref_out.ravel() - ov_out.ravel()).max())
print(f"ORT:      {ref_out.ravel()[:4]}")
print(f"OpenVINO: {ov_out.ravel()[:4]}")
print(f"max_abs={max_abs:.4f}")

if max_abs > 0.05:
    print(f"BUG REPRODUCED: OpenVINO expm1_bounded (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
