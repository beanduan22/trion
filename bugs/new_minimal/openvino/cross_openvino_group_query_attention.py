#!/usr/bin/env python3
"""
Bug: OpenVINO CPU MatMul+Reshape precision differs from ORT (group_query_attention)
Compiler: OpenVINO
Oracle:   ORT_DISABLE_ALL

Root cause: MatMul projects input to a wide embedding (inner dim 512) then
Reshape splits into multi-head format. OV tiles the MatMul differently and
the error accumulates before the reshape.

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
x = np.random.randn(2, 16).astype(np.float32)
W = np.random.randn(16, 512).astype(np.float32)
s = np.array([2, 8, 64], dtype=np.int64)   # reshape to [batch, heads, head_dim]

nodes = [
    oh.make_node("MatMul",  ["X","W"],  ["mm"]),
    oh.make_node("Reshape", ["mm","s"], ["Y"]),
]
graph = oh.make_graph(nodes, "gqa",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,  16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2, 8, 64])],
    initializer=[onh.from_array(W,"W"), onh.from_array(s,"s")])
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

if max_abs > 0.01:
    print(f"BUG REPRODUCED: OpenVINO group_query_attention (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
