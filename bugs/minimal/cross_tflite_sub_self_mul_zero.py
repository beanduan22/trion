#!/usr/bin/env python3
"""
Bug: TFLite diverges on sub_self_mul_zero
Compiler: TFLite
Oracle:   ORT_DISABLE_ALL
Patterns: Sub(x, x) -> MatMul should give zero
Root cause: TFLite may not preserve exact zero for sub_self.
Tolerance: 1e-4

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
x = np.random.randn(2, 64).astype(np.float32)
W = np.random.randn(64, 32).astype(np.float32) * 100.0

nodes = [
    oh.make_node("Sub",    ["X","X"],    ["zero"]),
    oh.make_node("MatMul", ["zero","W"], ["Y"]),
]
graph = oh.make_graph(nodes, "sub_self",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,64])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,32])],
    initializer=[onh.from_array(W,"W")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT (should be ~0): {ref.ravel()[:4]}")

try:
    import tempfile, os
    W_tf = tf.constant(W)
    @tf.function(input_signature=[tf.TensorSpec([2,64], tf.float32)])
    def sub_self_fn(x_in):
        zero = x_in - x_in
        return tf.matmul(zero, W_tf)

    with tempfile.TemporaryDirectory() as tmpdir:
        sm_path = os.path.join(tmpdir, "sm")
        tf.saved_model.save(sub_self_fn, sm_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
        tflite_model = converter.convert()
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        tfl_out = interp.get_tensor(out['index'])

    max_abs = float(np.abs(ref.ravel() - tfl_out.ravel()).max())
    tfl_max = float(np.abs(tfl_out).max())
    print(f"TFLite: {tfl_out.ravel()[:4]}")
    print(f"max_abs={max_abs:.6f}, tfl_max={tfl_max:.6f}")
    if max_abs > 1e-4 or tfl_max > 1e-4:
        print(f"BUG REPRODUCED: TFLite sub_self_mul_zero non-zero (max_abs={max_abs:.6f})")
        sys.exit(0)
    print("NOT reproduced"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED (TFLite error): {type(e).__name__}: {str(e)[:150]}")
    sys.exit(0)
