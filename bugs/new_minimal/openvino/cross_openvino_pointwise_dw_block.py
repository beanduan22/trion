#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_pointwise_dw_block
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Conv(1x1 pointwise) -> Conv(3x3 depthwise) -> Conv(1x1 pointwise)
Root cause : OV CPU depthwise Conv fp32 tiling differs from ORT.
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
W1 = np.random.randn(64, 32, 1, 1).astype(np.float32)   # expand
Wd = np.random.randn(64, 1, 3, 3).astype(np.float32)    # depthwise
W2 = np.random.randn(32, 64, 1, 1).astype(np.float32)   # project

nodes = [
    oh.make_node("Conv", ["X","W1"], ["pw1"],  kernel_shape=[1,1]),
    oh.make_node("Conv", ["pw1","Wd"], ["dw"], kernel_shape=[3,3], pads=[1,1,1,1], group=64),
    oh.make_node("Conv", ["dw","W2"], ["Y"],   kernel_shape=[1,1]),
]
graph = oh.make_graph(nodes, "dw_block",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,32,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,32,16,16])],
    initializer=[onh.from_array(W1,"W1"), onh.from_array(Wd,"Wd"), onh.from_array(W2,"W2")])
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
    print(f"BUG REPRODUCED: OpenVINO pointwise_dw_block (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
