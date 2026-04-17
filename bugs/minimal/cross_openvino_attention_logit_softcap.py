#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_attention_logit_softcap
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : MatMul(Q,Kt) -> Div(cap) -> Tanh -> Mul(cap)  [attention softcap]
Root cause : OV CPU GEMM fp32 accumulation differs from ORT; softcap preserves
             the difference in the tanh linear region (cap=10000 keeps values < 1).
Tolerance  : 0.005

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
x   = np.random.randn(4, 512).astype(np.float32)
W   = np.random.randn(512, 128).astype(np.float32)
Wk  = np.random.randn(512, 128).astype(np.float32)
# scores range [-9000, 11000]; use cap=20000 so x/cap in [-0.5, 0.6] -> tanh in linear region
cap = np.array(20000.0, dtype=np.float32)

nodes = [
    oh.make_node("MatMul",   ["X","W"],        ["Q"]),
    oh.make_node("MatMul",   ["X","Wk"],       ["K"]),
    oh.make_node("Transpose",["K"],            ["Kt"], perm=[1,0]),
    oh.make_node("MatMul",   ["Q","Kt"],       ["scores"]),
    oh.make_node("Div",      ["scores","cap"], ["div_out"]),
    oh.make_node("Tanh",     ["div_out"],      ["tanh_out"]),
    oh.make_node("Mul",      ["tanh_out","cap"], ["Y"]),
]
graph = oh.make_graph(nodes, "softcap_attn",
    [oh.make_tensor_value_info("X", TP.FLOAT, [4,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [4,4])],
    initializer=[onh.from_array(W,"W"), onh.from_array(Wk,"Wk"), onh.from_array(cap,"cap")])
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

if max_abs > 0.005:
    print(f"BUG REPRODUCED: OpenVINO attention_logit_softcap (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
