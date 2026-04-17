#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_redundant_reshape
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : MatMul(x,W) -> Reshape -> Reshape (redundant) -> output
Root cause : OV may eliminate redundant reshapes incorrectly, or GEMM fp32
             accumulation differs from ORT.
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
x   = np.random.randn(2, 512).astype(np.float32)
W   = np.random.randn(512, 64).astype(np.float32)
sh1 = np.array([1, 128], dtype=np.int64)
sh2 = np.array([2, 64],  dtype=np.int64)

nodes = [
    oh.make_node("MatMul",  ["X","W"],    ["mm"]),
    oh.make_node("Reshape", ["mm","sh1"], ["r1"]),
    oh.make_node("Reshape", ["r1","sh2"], ["Y"]),
]
graph = oh.make_graph(nodes, "redund_reshape",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[onh.from_array(W,"W"), onh.from_array(sh1,"sh1"), onh.from_array(sh2,"sh2")])
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
    print(f"BUG REPRODUCED: OpenVINO redundant_reshape (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
