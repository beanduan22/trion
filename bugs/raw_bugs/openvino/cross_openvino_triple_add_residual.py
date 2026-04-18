#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_triple_add_residual
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : MatMul(x,W) -> Add(r1) -> Add(r2) -> Add(r3)  [triple residual adds]
Root cause : OV CPU GEMM fp32 accumulation differs from ORT.
Tolerance  : 0.01

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
x  = np.random.randn(2, 512).astype(np.float32)
W  = np.random.randn(512, 64).astype(np.float32)
r1 = np.random.randn(2, 64).astype(np.float32)
r2 = np.random.randn(2, 64).astype(np.float32)
r3 = np.random.randn(2, 64).astype(np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W"],     ["mm"]),
    oh.make_node("Add",    ["mm","r1"],   ["a1"]),
    oh.make_node("Add",    ["a1","r2"],   ["a2"]),
    oh.make_node("Add",    ["a2","r3"],   ["Y"]),
]
graph = oh.make_graph(nodes, "triple_add",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[
        onh.from_array(W,"W"), onh.from_array(r1,"r1"),
        onh.from_array(r2,"r2"), onh.from_array(r3,"r3"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

core = ov.Core()
comp = core.compile_model(core.read_model(mb, b""), "CPU")
ov_out = np.array(comp({"X": x})[comp.output(0)])

max_abs = float(np.abs(ref.ravel() - ov_out.ravel()).max())
print(f"ORT:      {ref.ravel()[:4]}")
print(f"OpenVINO: {ov_out.ravel()[:4]}")
print(f"max_abs={max_abs:.4f}")

if max_abs > 0.01:
    print(f"BUG REPRODUCED: OpenVINO triple_add_residual (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
