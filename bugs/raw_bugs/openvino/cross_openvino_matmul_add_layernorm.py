#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_matmul_add_layernorm
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : MatMul(x, W) -> Add(bias) -> LayerNorm
Root cause : OV CPU GEMM tiles the inner-dimension sum differently from ORT,
             producing different fp32 rounding; LayerNorm amplifies the diff.
Tolerance  : 0.005

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
x  = np.random.randn(4, 512).astype(np.float32)
W  = np.random.randn(512, 128).astype(np.float32)
b  = np.random.randn(128).astype(np.float32)
ln_scale = np.ones(128, dtype=np.float32)
ln_bias  = np.zeros(128, dtype=np.float32)

nodes = [
    oh.make_node("MatMul",             ["X", "W"],           ["mm"]),
    oh.make_node("Add",                ["mm", "b"],          ["add"]),
    oh.make_node("LayerNormalization", ["add", "ls", "lb"], ["Y"],
                 axis=-1, epsilon=1e-5),
]
graph = oh.make_graph(nodes, "test",
    [oh.make_tensor_value_info("X",  TP.FLOAT, [4, 512])],
    [oh.make_tensor_value_info("Y",  TP.FLOAT, [4, 128])],
    initializer=[
        onh.from_array(W,        "W"),
        onh.from_array(b,        "b"),
        onh.from_array(ln_scale, "ls"),
        onh.from_array(ln_bias,  "lb"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so,
    providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

core = ov.Core()
comp = core.compile_model(core.read_model(mb, b""), "CPU")
ov_out = np.array(comp({"X": x})[comp.output(0)])

max_abs = float(np.abs(ref.ravel() - ov_out.ravel()).max())
print(f"ORT:      {ref.ravel()[:4]}")
print(f"OpenVINO: {ov_out.ravel()[:4]}")
print(f"max_abs={max_abs:.4f}")

if max_abs > 0.005:
    print(f"BUG REPRODUCED: OpenVINO matmul_add_layernorm (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
