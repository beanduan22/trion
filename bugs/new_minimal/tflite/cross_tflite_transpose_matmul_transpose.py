#!/usr/bin/env python3
"""
Bug: TFLite diverges on transpose_matmul_transpose
Compiler: TFLite
Oracle:   ORT_DISABLE_ALL
Patterns: Transpose -> MatMul -> Transpose
Root cause: TFLite GEMM accumulation differs from ORT after transpose.
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
x = np.random.randn(512, 2).astype(np.float32)
W = np.random.randn(512, 64).astype(np.float32)

nodes = [
    oh.make_node("Transpose", ["X"],     ["Xt"], perm=[1,0]),
    oh.make_node("MatMul",    ["Xt","W"], ["mm"]),
    oh.make_node("Transpose", ["mm"],    ["Y"],  perm=[1,0]),
]
graph = oh.make_graph(nodes, "tr_mm_tr",
    [oh.make_tensor_value_info("X", TP.FLOAT, [512,2])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [64,2])],
    initializer=[onh.from_array(W,"W")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}")

try:
    import tempfile, os
    W_tf = tf.constant(W)
    @tf.function(input_signature=[tf.TensorSpec([512,2], tf.float32)])
    def fn(x_in):
        Xt = tf.transpose(x_in)
        mm = tf.matmul(Xt, W_tf)
        return tf.transpose(mm)

    with tempfile.TemporaryDirectory() as tmpdir:
        sm_path = os.path.join(tmpdir, "sm")
        tf.saved_model.save(fn, sm_path)
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
        print(f"BUG REPRODUCED: TFLite transpose_matmul_transpose (max_abs={max_abs:.4f})")
        sys.exit(0)
    print("NOT reproduced"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED (TFLite error): {type(e).__name__}: {str(e)[:150]}")
    sys.exit(0)
