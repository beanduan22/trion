#!/usr/bin/env python3
"""
Bug ID     : github_ort_005
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : layernorm fusion wrong output
Root cause : Bug: FusionLayerNormalization changes output for non-standard shapes.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Bug: FusionLayerNormalization changes output for non-standard shapes.
# ORT_ENABLE_ALL fuses the manual pattern; result differs from unfused.
np.random.seed(7)
batch, seq, hidden = 2, 4, 8
x     = np.random.randn(batch, seq, hidden).astype(np.float32)
scale = np.ones(hidden, dtype=np.float32)
bias  = np.zeros(hidden, dtype=np.float32)
eps   = np.array(1e-5, dtype=np.float32)
two   = np.array(2.0,  dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch, seq, hidden])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, seq, hidden])
inits = [
    numpy_helper.from_array(scale, "scale"),
    numpy_helper.from_array(bias,  "bias"),
    numpy_helper.from_array(eps,   "eps"),
    numpy_helper.from_array(two,   "two"),
]
nodes = [
    helper.make_node("ReduceMean", ["X"],          ["mean"],    axes=[-1], keepdims=1),
    helper.make_node("Sub",        ["X", "mean"],  ["diff"]),
    helper.make_node("Pow",        ["diff", "two"],["sq"]),
    helper.make_node("ReduceMean", ["sq"],         ["var"],     axes=[-1], keepdims=1),
    helper.make_node("Add",        ["var", "eps"], ["var_eps"]),
    helper.make_node("Sqrt",       ["var_eps"],    ["std"]),
    helper.make_node("Div",        ["diff", "std"],["norm"]),
    helper.make_node("Mul",        ["norm", "scale"], ["scaled"]),
    helper.make_node("Add",        ["scaled", "bias"], ["Y"]),
]
graph = helper.make_graph(nodes, "manual_ln", [X], [Y], initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

so_all = ort.SessionOptions()
so_all.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so_basic = ort.SessionOptions()
so_basic.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

out_all   = ort.InferenceSession(model.SerializeToString(), so_all,
                                 providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
out_basic = ort.InferenceSession(model.SerializeToString(), so_basic,
                                 providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

max_diff = float(np.max(np.abs(out_all - out_basic)))
print(f"ORT_ENABLE_ALL  [0,0,:3]: {out_all[0,0,:3]}")
print(f"ORT_ENABLE_BASIC [0,0,:3]: {out_basic[0,0,:3]}")
print(f"Max diff fused vs unfused: {max_diff:.6f}")
PASS = (max_diff < 1e-4)
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
