#!/usr/bin/env python3
"""
Bug: OpenVINO CPU Conv+PReLU (rrelu_inference_identity) precision differs from ORT
Compiler: OpenVINO
Oracle:   ORT_DISABLE_ALL

Root cause: In inference mode RReLU reduces to PReLU with a fixed slope.
OV CPU applies Winograd/tiled GEMM to the Conv layer and then fuses the
PReLU activation, reordering the fp32 accumulation. This introduces a
precision gap of ~0.12 max absolute error vs ORT's sequential reference.

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
C = 32
x     = np.random.randn(1, C, 8, 8).astype(np.float32)
W     = np.random.randn(C, C, 3, 3).astype(np.float32)
slope = np.full((C, 1, 1), 0.25, dtype=np.float32)   # fixed slope = RReLU inference mode

nodes = [
    oh.make_node("Conv",  ["X","W"],       ["h"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("PRelu", ["h","slope"],   ["Y"]),
]
graph = oh.make_graph(nodes, "rrelu_inf",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,C,8,8])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,C,8,8])],
    initializer=[onh.from_array(W,"W"), onh.from_array(slope,"slope")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref_out = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

core = ov.Core()
comp = core.compile_model(core.read_model(mb, b""), "CPU")
ov_out = np.array(comp({"X": x})[comp.output(0)])

max_abs = float(np.abs(ref_out.ravel() - ov_out.ravel()).max())
print(f"ORT:      {ref_out.ravel()[:4]}")
print(f"OpenVINO: {ov_out.ravel()[:4]}")
print(f"max_abs={max_abs:.4f}")

if max_abs > 0.05:
    print(f"BUG REPRODUCED: OpenVINO rrelu_inference_identity (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
