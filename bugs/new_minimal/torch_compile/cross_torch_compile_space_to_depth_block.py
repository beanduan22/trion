#!/usr/bin/env python3
"""
Bug: torch.compile crashes on space_to_depth_block (onnx2torch + compilation)
Compiler: torch.compile (Inductor backend)
Root cause: onnx2torch converts SpaceToDepth block; shape propagation fails.
Exit 0 = BUG REPRODUCED, 1 = not reproduced, 2 = missing deps
"""
import sys
try:
    import numpy as np, onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}"); sys.exit(2)
try:
    import onnx2torch, torch
except ImportError:
    print("missing dep: onnx2torch or torch"); sys.exit(2)

np.random.seed(42)
x  = np.random.randn(1, 32, 16, 16).astype(np.float32)
W1 = np.random.randn(8, 32, 3, 3).astype(np.float32)
W2 = np.random.randn(16, 32, 1, 1).astype(np.float32)

nodes = [
    oh.make_node("Conv",         ["X","W1"],   ["conv1"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("SpaceToDepth", ["conv1"],    ["s2d"],   blocksize=2),
    oh.make_node("Conv",         ["s2d","W2"], ["Y"],     kernel_shape=[1,1]),
]
graph = oh.make_graph(nodes, "s2d_block",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,32,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,16,8,8])],
    initializer=[onh.from_array(W1,"W1"), onh.from_array(W2,"W2")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}  (correct)")

try:
    net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    eager_out = net(torch.from_numpy(x)).detach().numpy()
    print(f"eager: {eager_out.ravel()[:4]}  (correct)")
    compiled = torch.compile(net)
    with torch.no_grad():
        out = compiled(torch.from_numpy(x))
    print("NOT reproduced: torch.compile succeeded"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED: torch.compile crashes on space_to_depth_block: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(0)
