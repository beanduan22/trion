#!/usr/bin/env python3
"""
Bug: TFLite diverges on transformer_encoder_layer
Compiler: TFLite
Oracle:   ORT_DISABLE_ALL
Patterns: Self-attention + FFN transformer pattern
Root cause: TFLite fp32 GEMM accumulation differs from ORT in complex transformer.
Tolerance: 0.01

Exit 0 = BUG REPRODUCED, 1 = not reproduced, 2 = missing deps
"""
import sys
try:
    import numpy as np, onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}"); sys.exit(2)
try:
    import tensorflow as tf
except ImportError:
    print("missing dep: tensorflow"); sys.exit(2)

np.random.seed(42)
S, D = 2, 64
x    = np.random.randn(S, 512).astype(np.float32)
Wq   = np.random.randn(512, D).astype(np.float32)
Wk   = np.random.randn(512, D).astype(np.float32)
Wv   = np.random.randn(512, D).astype(np.float32)
Wff1 = np.random.randn(D, D*4).astype(np.float32)
Wff2 = np.random.randn(D*4, D).astype(np.float32)
Wx   = np.random.randn(512, D).astype(np.float32)
scale = (D**-0.5)

# Build and run ORT reference via ONNX
nodes = [
    oh.make_node("MatMul",   ["X","Wx"],      ["xp"]),
    oh.make_node("MatMul",   ["X","Wq"],      ["Q"]),
    oh.make_node("MatMul",   ["X","Wk"],      ["K"]),
    oh.make_node("MatMul",   ["X","Wv"],      ["V"]),
    oh.make_node("Transpose",["K"],           ["Kt"], perm=[1,0]),
    oh.make_node("MatMul",   ["Q","Kt"],      ["scores"]),
    oh.make_node("Mul",      ["scores","sc"], ["scores_s"]),
    oh.make_node("Softmax",  ["scores_s"],    ["attn"], axis=-1),
    oh.make_node("MatMul",   ["attn","V"],    ["ctx"]),
    oh.make_node("Add",      ["xp","ctx"],    ["res1"]),
    oh.make_node("MatMul",   ["res1","Wff1"], ["ff1"]),
    oh.make_node("Relu",     ["ff1"],         ["ff1_r"]),
    oh.make_node("MatMul",   ["ff1_r","Wff2"], ["ff2"]),
    oh.make_node("Add",      ["res1","ff2"],  ["Y"]),
]
sc = np.array(scale, dtype=np.float32)
graph = oh.make_graph(nodes, "transformer",
    [oh.make_tensor_value_info("X", TP.FLOAT, [S,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [S,D])],
    initializer=[
        onh.from_array(Wx,"Wx"), onh.from_array(Wq,"Wq"), onh.from_array(Wk,"Wk"),
        onh.from_array(Wv,"Wv"), onh.from_array(sc,"sc"),
        onh.from_array(Wff1,"Wff1"), onh.from_array(Wff2,"Wff2"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}")

try:
    import tempfile, os, math
    Wx_tf = tf.constant(Wx); Wq_tf = tf.constant(Wq); Wk_tf = tf.constant(Wk)
    Wv_tf = tf.constant(Wv); Wff1_tf = tf.constant(Wff1); Wff2_tf = tf.constant(Wff2)
    sc_tf = tf.constant(scale, dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([S,512], tf.float32)])
    def transformer_fn(x_in):
        xp = tf.matmul(x_in, Wx_tf)
        Q = tf.matmul(x_in, Wq_tf)
        K = tf.matmul(x_in, Wk_tf)
        V = tf.matmul(x_in, Wv_tf)
        Kt = tf.transpose(K)
        scores = tf.matmul(Q, Kt) * sc_tf
        attn = tf.nn.softmax(scores, axis=-1)
        ctx = tf.matmul(attn, V)
        res1 = xp + ctx
        ff1 = tf.nn.relu(tf.matmul(res1, Wff1_tf))
        ff2 = tf.matmul(ff1, Wff2_tf)
        return res1 + ff2

    with tempfile.TemporaryDirectory() as tmpdir:
        sm_path = os.path.join(tmpdir, "sm")
        tf.saved_model.save(transformer_fn, sm_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
        tflite_model = converter.convert()
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        tfl_out = interp.get_tensor(out['index'])

    max_abs = float(np.abs(ref.ravel() - tfl_out.ravel()).max())
    print(f"TFLite: {tfl_out.ravel()[:4]}")
    print(f"max_abs={max_abs:.4f}")
    if max_abs > 0.01:
        print(f"BUG REPRODUCED: TFLite transformer_encoder_layer (max_abs={max_abs:.4f})")
        sys.exit(0)
    print("NOT reproduced"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED (TFLite error): {type(e).__name__}: {str(e)[:150]}")
    sys.exit(0)
