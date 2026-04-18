#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_flex_attention_precision_sdpa
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Scaled Dot-Product Attention: MatMul(Q,Kt)*scale -> Softmax -> MatMul(xV)
Root cause : OV CPU GEMM fp32 inner-dim accumulation differs in attention path.
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
x     = np.random.randn(2, 512).astype(np.float32)
Wq    = np.random.randn(512, 64).astype(np.float32)
Wk    = np.random.randn(512, 64).astype(np.float32)
Wv    = np.random.randn(512, 64).astype(np.float32)
scale = np.array(64**-0.5, dtype=np.float32)

nodes = [
    oh.make_node("MatMul",   ["X","Wq"],      ["Q"]),
    oh.make_node("MatMul",   ["X","Wk"],      ["K"]),
    oh.make_node("MatMul",   ["X","Wv"],      ["V"]),
    oh.make_node("Transpose",["K"],           ["Kt"], perm=[1,0]),
    oh.make_node("MatMul",   ["Q","Kt"],      ["scores"]),
    oh.make_node("Mul",      ["scores","sc"], ["scores_s"]),
    oh.make_node("Softmax",  ["scores_s"],    ["attn"], axis=-1),
    oh.make_node("MatMul",   ["attn","V"],    ["Y"]),
]
graph = oh.make_graph(nodes, "sdpa",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[
        onh.from_array(Wq,"Wq"), onh.from_array(Wk,"Wk"), onh.from_array(Wv,"Wv"),
        onh.from_array(scale,"sc"),
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
    print(f"BUG REPRODUCED: OpenVINO flex_attention_precision_sdpa (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
