#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_conv_prelu_channel
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Conv(3x3) -> PRelu(per-channel slope)
Root cause : OV CPU Conv fp32 tiling differs; PRelu channels expose.
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
x     = np.random.randn(1, 32, 16, 16).astype(np.float32)
W     = np.random.randn(64, 32, 3, 3).astype(np.float32)
slope = np.random.uniform(0.01, 0.3, (64,1,1)).astype(np.float32)

nodes = [
    oh.make_node("Conv",  ["X","W"],      ["conv"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("PRelu", ["conv","slope"], ["Y"]),
]
graph = oh.make_graph(nodes, "test",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,32,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,64,16,16])],
    initializer=[onh.from_array(W,"W"), onh.from_array(slope,"slope")])
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
    print(f"BUG REPRODUCED: OpenVINO conv_prelu_channel (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
