#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_conv_bn_fusion
Source     : Cross-compiler testing (2026-04-13)
Compiler   : OpenVINO 2026.0
Patterns   : Conv(3x3, pad=1) + BatchNormalization(eps=1e-5)
Root cause : OpenVINO Conv+BN fusion introduces rounding error of 0.014 vs ORT unoptimized reference when BN has non-trivial scale (0.5-2.0) and non-zero mean.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys

try:
    import numpy as np
    import onnx
    from onnx import helper as oh, TensorProto, numpy_helper as onh
    import onnxruntime as ort
    import openvino as ov
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

np.random.seed(42)

C_IN, C_OUT = 4, 8
x = np.random.randn(1, C_IN, 8, 8).astype(np.float32)
conv_w = np.random.randn(C_OUT, C_IN, 3, 3).astype(np.float32) * 0.1
conv_b = np.zeros(C_OUT, dtype=np.float32)
bn_scale = np.random.uniform(0.5, 2.0, C_OUT).astype(np.float32)
bn_bias = np.random.uniform(-1.0, 1.0, C_OUT).astype(np.float32)
bn_mean = np.random.uniform(-0.5, 0.5, C_OUT).astype(np.float32)
bn_var = np.random.uniform(0.1, 1.0, C_OUT).astype(np.float32)

nodes = [
    oh.make_node("Conv", ["X", "cw", "cb"], ["conv_out"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    oh.make_node("BatchNormalization",
        ["conv_out", "bs", "bb", "bm", "bv"], ["Y"], epsilon=1e-5),
]
graph = oh.make_graph(nodes, "test",
    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, C_IN, 8, 8])],
    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, C_OUT, 8, 8])],
    initializer=[
        onh.from_array(conv_w, "cw"), onh.from_array(conv_b, "cb"),
        onh.from_array(bn_scale, "bs"), onh.from_array(bn_bias, "bb"),
        onh.from_array(bn_mean, "bm"), onh.from_array(bn_var, "bv"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
model_bytes = model.SerializeToString()

# ORT reference (no optimizations)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(model_bytes, sess_options=so,
    providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

# OpenVINO
core = ov.Core()
ov_model = core.read_model(model_bytes, b'')
compiled = core.compile_model(ov_model, "CPU")
ov_out = compiled({"X": x})[compiled.output(0)]

diff = float(np.max(np.abs(ov_out.astype(np.float64) - ref.astype(np.float64))))
print(f"ORT ref[:4]:      {ref.ravel()[:4]}")
print(f"OpenVINO[:4]:     {ov_out.ravel()[:4]}")
print(f"max_diff={diff:.6f}  tol=0.01")

PASS = diff <= 0.01
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
