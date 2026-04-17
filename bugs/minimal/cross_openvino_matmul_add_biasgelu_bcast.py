#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_matmul_add_biasgelu_bcast
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : MatMul(x,W) -> Add(bias) -> GELU -> Mul(gate)  [biased GELU gate]
Root cause : OV CPU GEMM fp32 accumulation differs; GELU nonlinearity amplifies.
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
x    = np.random.randn(2, 512).astype(np.float32)
W    = np.random.randn(512, 64).astype(np.float32)
b    = np.random.randn(64).astype(np.float32)
gate = np.random.randn(2, 64).astype(np.float32)

# Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x+0.044715*x^3)))
# Build as: mm+b -> x*0.5*(1+tanh(c1*(x+c2*x^3)))
c1 = np.array(0.7978845608, dtype=np.float32)
c2 = np.array(0.044715, dtype=np.float32)
c05 = np.array(0.5, dtype=np.float32)
c1v = np.array(1.0, dtype=np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W"],        ["mm"]),
    oh.make_node("Add",    ["mm","b"],        ["h"]),
    # GELU approx
    oh.make_node("Mul",    ["h","h"],         ["h2"]),
    oh.make_node("Mul",    ["h2","h"],        ["h3"]),
    oh.make_node("Mul",    ["h3","c2"],       ["h3c"]),
    oh.make_node("Add",    ["h","h3c"],       ["hsum"]),
    oh.make_node("Mul",    ["hsum","c1"],     ["harg"]),
    oh.make_node("Tanh",   ["harg"],          ["htanh"]),
    oh.make_node("Add",    ["htanh","c1v"],   ["h1p"]),
    oh.make_node("Mul",    ["h1p","c05"],     ["hgelu"]),
    oh.make_node("Mul",    ["h","hgelu"],     ["gelu_out"]),
    oh.make_node("Mul",    ["gelu_out","G"],  ["Y"]),
]
graph = oh.make_graph(nodes, "biasgelu",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[
        onh.from_array(W,"W"), onh.from_array(b,"b"), onh.from_array(gate,"G"),
        onh.from_array(c1,"c1"), onh.from_array(c2,"c2"),
        onh.from_array(c05,"c05"), onh.from_array(c1v,"c1v"),
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
    print(f"BUG REPRODUCED: OpenVINO matmul_add_biasgelu_bcast (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
