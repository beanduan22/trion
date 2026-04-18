#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_inception_v3_branch
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Multiple Conv branches (1x1, 3x3, 5x5) -> Concat  [Inception module]
Root cause : OV CPU Conv fp32 tiling differs from ORT; multi-branch accumulates.
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
x  = np.random.randn(1, 32, 16, 16).astype(np.float32)
W1 = np.random.randn(32, 32, 1, 1).astype(np.float32)
W3 = np.random.randn(32, 32, 3, 3).astype(np.float32)
W5 = np.random.randn(16, 32, 5, 5).astype(np.float32)
Wp = np.random.randn(16, 32, 1, 1).astype(np.float32)

nodes = [
    oh.make_node("Conv", ["X","W1"], ["b1"], kernel_shape=[1,1]),
    oh.make_node("Conv", ["X","W3"], ["b3"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("Conv", ["X","W5"], ["b5"], kernel_shape=[5,5], pads=[2,2,2,2]),
    oh.make_node("MaxPool", ["X"],   ["pool"], kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]),
    oh.make_node("Conv", ["pool","Wp"], ["bp"], kernel_shape=[1,1]),
    oh.make_node("Concat", ["b1","b3","b5","bp"], ["Y"], axis=1),
]
graph = oh.make_graph(nodes, "inception",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,32,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,96,16,16])],
    initializer=[
        onh.from_array(W1,"W1"), onh.from_array(W3,"W3"),
        onh.from_array(W5,"W5"), onh.from_array(Wp,"Wp"),
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
    print(f"BUG REPRODUCED: OpenVINO inception_v3_branch (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
