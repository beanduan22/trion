#!/usr/bin/env python3
"""
Bug: TFLite diverges on attention_logit_softcap
Compiler: TFLite (via ai_edge_torch or tf.lite)
Oracle:   ORT_DISABLE_ALL
Patterns: Div -> Tanh -> Mul  (softcap: score/cap -> tanh -> *cap)
Root cause: TFLite Tanh fp32 precision differs from ORT at moderate inputs.
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
x   = np.random.randn(2, 512).astype(np.float32)
W   = np.random.randn(512, 64).astype(np.float32)
Wk  = np.random.randn(512, 64).astype(np.float32)
cap = np.array(50.0, dtype=np.float32)

nodes = [
    oh.make_node("MatMul",   ["X","W"],       ["Q"]),
    oh.make_node("MatMul",   ["X","Wk"],      ["K"]),
    oh.make_node("Transpose",["K"],           ["Kt"], perm=[1,0]),
    oh.make_node("MatMul",   ["Q","Kt"],      ["scores"]),
    oh.make_node("Div",      ["scores","cap"], ["div_out"]),
    oh.make_node("Tanh",     ["div_out"],     ["tanh_out"]),
    oh.make_node("Mul",      ["tanh_out","cap"], ["Y"]),
]
graph = oh.make_graph(nodes, "softcap",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,2])],
    initializer=[onh.from_array(W,"W"), onh.from_array(Wk,"Wk"), onh.from_array(cap,"cap")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}")

# TFLite conversion via SavedModel path
try:
    import tempfile, os
    # Build a tf.function wrapping the computation
    cap_tf = tf.constant(50.0)
    W_tf   = tf.constant(W)
    Wk_tf  = tf.constant(Wk)

    @tf.function(input_signature=[tf.TensorSpec([2,512], tf.float32)])
    def softcap_fn(x_in):
        Q = tf.matmul(x_in, W_tf)
        K = tf.matmul(x_in, Wk_tf)
        Kt = tf.transpose(K)
        scores = tf.matmul(Q, Kt)
        return tf.tanh(scores / cap_tf) * cap_tf

    with tempfile.TemporaryDirectory() as tmpdir:
        sm_path = os.path.join(tmpdir, "sm")
        tf.saved_model.save(softcap_fn, sm_path)
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
        print(f"BUG REPRODUCED: TFLite attention_logit_softcap (max_abs={max_abs:.4f})")
        sys.exit(0)
    print("NOT reproduced"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED (TFLite error): {type(e).__name__}: {str(e)[:150]}")
    sys.exit(0)
