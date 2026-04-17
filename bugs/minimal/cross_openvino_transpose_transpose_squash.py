#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_transpose_transpose_squash
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Transpose([0,2,3,1]) -> Transpose([0,3,1,2]) -> MatMul
             Two transposes that compose to identity — OV may squash incorrectly.
Root cause : OV optimizer may incorrectly squash two consecutive transposes,
             producing wrong output order.
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
# [2,4,8,16] -> perm[0,2,3,1] -> [2,8,16,4] -> perm[0,3,1,2] -> [2,4,8,16]
x = np.random.randn(2, 4, 8, 16).astype(np.float32)
# After double-transpose we get back [2,4,8,16]
# Reshape to [8, 128] then MatMul [128, 8] -> [8, 8]
W = np.random.randn(128, 8).astype(np.float32)
sh = np.array([8, 128], dtype=np.int64)  # 2*4=8, 8*16=128

nodes = [
    oh.make_node("Transpose", ["X"],    ["t1"], perm=[0,2,3,1]),  # [2,8,16,4]
    oh.make_node("Transpose", ["t1"],   ["t2"], perm=[0,3,1,2]),  # [2,4,8,16]
    oh.make_node("Reshape",   ["t2","sh"], ["flat"]),              # [8, 128]
    oh.make_node("MatMul",    ["flat","W"], ["Y"]),                # [8, 8]
]
graph = oh.make_graph(nodes, "tr_squash",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,4,8,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [8,8])],
    initializer=[onh.from_array(W,"W"), onh.from_array(sh,"sh")])
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
    print(f"BUG REPRODUCED: OpenVINO transpose_transpose_squash (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
