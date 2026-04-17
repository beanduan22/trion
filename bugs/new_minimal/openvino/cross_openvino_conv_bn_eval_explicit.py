#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_conv_bn_eval_explicit
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Conv(3x3) -> BatchNormalization (eval mode, explicit params)
Root cause : OV CPU Conv+BN fusion fp32 rounding differs from ORT unoptimized.
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
C_IN, C_OUT = 32, 64
x      = np.random.randn(1, C_IN, 16, 16).astype(np.float32)
W      = np.random.randn(C_OUT, C_IN, 3, 3).astype(np.float32)
b      = np.zeros(C_OUT, dtype=np.float32)
scale  = np.random.uniform(0.5, 2.0, C_OUT).astype(np.float32)
bshift = np.random.uniform(-1.0, 1.0, C_OUT).astype(np.float32)
mean   = np.random.uniform(-0.5, 0.5, C_OUT).astype(np.float32)
var    = np.random.uniform(0.1, 1.0, C_OUT).astype(np.float32)

nodes = [
    oh.make_node("Conv", ["X","W","cb"], ["conv"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("BatchNormalization", ["conv","sc","bs","bm","bv"], ["Y"], epsilon=1e-5),
]
graph = oh.make_graph(nodes, "test",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,C_IN,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,C_OUT,16,16])],
    initializer=[
        onh.from_array(W,"W"), onh.from_array(b,"cb"),
        onh.from_array(scale,"sc"), onh.from_array(bshift,"bs"),
        onh.from_array(mean,"bm"), onh.from_array(var,"bv"),
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
    print(f"BUG REPRODUCED: OpenVINO conv_bn_eval_explicit (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
