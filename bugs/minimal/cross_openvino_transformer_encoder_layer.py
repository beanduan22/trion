#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_transformer_encoder_layer
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Self-attention + FFN: MatMul(Q,Kt) -> Softmax -> MatMul(xV) -> Add -> LN
             -> Gemm -> GELU -> Gemm -> Add -> LN
Root cause : OV CPU GEMM fp32 accumulation differs in transformer pattern.
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
S, D = 2, 64    # seq_len, d_model (small for minimal repro)
x    = np.random.randn(S, 512).astype(np.float32)
Wq   = np.random.randn(512, D).astype(np.float32)
Wk   = np.random.randn(512, D).astype(np.float32)
Wv   = np.random.randn(512, D).astype(np.float32)
Wo   = np.random.randn(D, D).astype(np.float32)
Wff1 = np.random.randn(D, D*4).astype(np.float32)
Wff2 = np.random.randn(D*4, D).astype(np.float32)
scale = np.array(D**-0.5, dtype=np.float32)
ln1_s = np.ones(D, dtype=np.float32); ln1_b = np.zeros(D, dtype=np.float32)
ln2_s = np.ones(D, dtype=np.float32); ln2_b = np.zeros(D, dtype=np.float32)
# GELU constants
c1 = np.array(0.7978845608, dtype=np.float32)
c2 = np.array(0.044715, dtype=np.float32)
c05 = np.array(0.5, dtype=np.float32); c1v = np.array(1.0, dtype=np.float32)
# x_proj: input projected to D
Wx = np.random.randn(512, D).astype(np.float32)

nodes = [
    # project input
    oh.make_node("MatMul",   ["X","Wx"],      ["xp"]),
    # attention
    oh.make_node("MatMul",   ["X","Wq"],      ["Q"]),
    oh.make_node("MatMul",   ["X","Wk"],      ["K"]),
    oh.make_node("MatMul",   ["X","Wv"],      ["V"]),
    oh.make_node("Transpose",["K"],           ["Kt"], perm=[1,0]),
    oh.make_node("MatMul",   ["Q","Kt"],      ["scores"]),
    oh.make_node("Mul",      ["scores","sc"], ["scores_s"]),
    oh.make_node("Softmax",  ["scores_s"],    ["attn"], axis=-1),
    oh.make_node("MatMul",   ["attn","V"],    ["ctx"]),
    oh.make_node("MatMul",   ["ctx","Wo"],    ["proj"]),
    # residual + LN
    oh.make_node("Add",      ["xp","proj"],   ["res1"]),
    oh.make_node("LayerNormalization", ["res1","ln1s","ln1b"], ["ln1"], axis=-1, epsilon=1e-5),
    # FFN
    oh.make_node("MatMul",   ["ln1","Wff1"],  ["ff1"]),
    oh.make_node("Mul",      ["ff1","ff1"],   ["ff1_2"]),
    oh.make_node("Mul",      ["ff1_2","ff1"], ["ff1_3"]),
    oh.make_node("Mul",      ["ff1_3","c2"],  ["ff1_3c"]),
    oh.make_node("Add",      ["ff1","ff1_3c"],["ffsum"]),
    oh.make_node("Mul",      ["ffsum","c1"],  ["ffarg"]),
    oh.make_node("Tanh",     ["ffarg"],       ["fftanh"]),
    oh.make_node("Add",      ["fftanh","c1v"],["ff1p"]),
    oh.make_node("Mul",      ["ff1p","c05"],  ["ffgelu"]),
    oh.make_node("Mul",      ["ff1","ffgelu"],["ff1_act"]),
    oh.make_node("MatMul",   ["ff1_act","Wff2"], ["ff2"]),
    # residual + LN
    oh.make_node("Add",      ["ln1","ff2"],   ["res2"]),
    oh.make_node("LayerNormalization", ["res2","ln2s","ln2b"], ["Y"], axis=-1, epsilon=1e-5),
]
graph = oh.make_graph(nodes, "transformer_enc",
    [oh.make_tensor_value_info("X", TP.FLOAT, [S,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [S,D])],
    initializer=[
        onh.from_array(Wx,"Wx"), onh.from_array(Wq,"Wq"), onh.from_array(Wk,"Wk"),
        onh.from_array(Wv,"Wv"), onh.from_array(Wo,"Wo"),
        onh.from_array(scale,"sc"),
        onh.from_array(Wff1,"Wff1"), onh.from_array(Wff2,"Wff2"),
        onh.from_array(ln1_s,"ln1s"), onh.from_array(ln1_b,"ln1b"),
        onh.from_array(ln2_s,"ln2s"), onh.from_array(ln2_b,"ln2b"),
        onh.from_array(c1,"c1"), onh.from_array(c2,"c2"),
        onh.from_array(c05,"c05"), onh.from_array(c1v,"c1v"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
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
    print(f"BUG REPRODUCED: OpenVINO transformer_encoder_layer (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
