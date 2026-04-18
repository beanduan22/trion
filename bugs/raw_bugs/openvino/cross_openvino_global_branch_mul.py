#!/usr/bin/env python3
"""
Bug: OpenVINO CPU Conv+branch+Mul precision differs from ORT (global_branch_mul)
Compiler: OpenVINO
Oracle:   ORT_DISABLE_ALL

Root cause: Two parallel Conv paths whose outputs are multiplied element-wise.
OV applies different fp32 GEMM tiling to each branch and the resulting Mul
of two independently-accumulated tensors compounds the precision error.
Max error typically 3–10× at C=32.

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
x  = np.random.randn(1, C, 16, 16).astype(np.float32)
W1 = np.random.randn(C, C, 3, 3).astype(np.float32)   # 3×3 branch
W2 = np.random.randn(C, C, 1, 1).astype(np.float32)   # 1×1 branch

nodes = [
    oh.make_node("Conv", ["X","W1"], ["b1"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("Conv", ["X","W2"], ["b2"], kernel_shape=[1,1]),
    oh.make_node("Mul",  ["b1","b2"],["Y"]),
]
graph = oh.make_graph(nodes, "global_branch_mul",
    [oh.make_tensor_value_info("X",  TP.FLOAT, [1,C,16,16])],
    [oh.make_tensor_value_info("Y",  TP.FLOAT, [1,C,16,16])],
    initializer=[onh.from_array(W1,"W1"), onh.from_array(W2,"W2")])
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

if max_abs > 0.5:
    print(f"BUG REPRODUCED: OpenVINO global_branch_mul (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
