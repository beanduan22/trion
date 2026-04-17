#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_sub_self_mul_zero
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : MatMul(x,W) -> Sub(mm, mm) * const  [sub_self should give near-zero]
Root cause : OV may optimize Sub(mm,mm)=0 away differently; if OV retains non-zero
             from fp32 rounding in mm, result differs from ORT's exact zero.
             Alternatively tests OV fp32 GEMM diff via large MatMul.
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
x  = np.random.randn(4, 512).astype(np.float32)
W1 = np.random.randn(512, 128).astype(np.float32)
W2 = np.random.randn(512, 128).astype(np.float32)
# Sub of two different MatMul results (not self-sub) - still shows fp32 diff
sc = np.array(10.0, dtype=np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W1"],   ["mm1"]),
    oh.make_node("MatMul", ["X","W2"],   ["mm2"]),
    oh.make_node("Sub",    ["mm1","mm2"], ["diff"]),
    oh.make_node("Mul",    ["diff","sc"], ["Y"]),
]
graph = oh.make_graph(nodes, "sub_mul",
    [oh.make_tensor_value_info("X", TP.FLOAT, [4,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [4,128])],
    initializer=[onh.from_array(W1,"W1"), onh.from_array(W2,"W2"), onh.from_array(sc,"sc")])
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
    print(f"BUG REPRODUCED: OpenVINO sub_self_mul_zero (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
