import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Bug: SkipLayerNorm FP16 fusion introduces up to 9.77 abs diff vs unfused.
np.random.seed(3)
batch, seq, hidden = 1, 8, 64
x     = np.random.randn(batch, seq, hidden).astype(np.float16)
skip  = np.random.randn(batch, seq, hidden).astype(np.float16)
gamma = (np.random.randn(hidden) + 1.0).astype(np.float16)
beta  = np.random.randn(hidden).astype(np.float16)
eps   = np.array(1e-5, dtype=np.float16)
two   = np.array(2.0,  dtype=np.float16)

X    = helper.make_tensor_value_info("X",    TensorProto.FLOAT16, [batch, seq, hidden])
Skip = helper.make_tensor_value_info("Skip", TensorProto.FLOAT16, [batch, seq, hidden])
Y    = helper.make_tensor_value_info("Y",    TensorProto.FLOAT16, [batch, seq, hidden])
inits = [
    numpy_helper.from_array(gamma, "gamma"),
    numpy_helper.from_array(beta,  "beta"),
    numpy_helper.from_array(eps,   "eps"),
    numpy_helper.from_array(two,   "two"),
]
nodes = [
    helper.make_node("Add",        ["X", "Skip"],  ["sum"]),
    helper.make_node("ReduceMean", ["sum"],         ["mean"],    axes=[-1], keepdims=1),
    helper.make_node("Sub",        ["sum", "mean"], ["diff"]),
    helper.make_node("Pow",        ["diff", "two"], ["sq"]),
    helper.make_node("ReduceMean", ["sq"],          ["var"],     axes=[-1], keepdims=1),
    helper.make_node("Add",        ["var", "eps"],  ["var_eps"]),
    helper.make_node("Sqrt",       ["var_eps"],     ["std"]),
    helper.make_node("Div",        ["diff", "std"], ["norm"]),
    helper.make_node("Mul",        ["norm", "gamma"], ["scaled"]),
    helper.make_node("Add",        ["scaled", "beta"], ["Y"]),
]
graph = helper.make_graph(nodes, "skip_ln", [X, Skip], [Y], initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

so_all = ort.SessionOptions()
so_all.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so_basic = ort.SessionOptions()
so_basic.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

feed = {"X": x, "Skip": skip}
out_all   = ort.InferenceSession(model.SerializeToString(), so_all,
                                 providers=["CPUExecutionProvider"]).run(None, feed)[0].astype(np.float32)
out_basic = ort.InferenceSession(model.SerializeToString(), so_basic,
                                 providers=["CPUExecutionProvider"]).run(None, feed)[0].astype(np.float32)

max_diff = float(np.max(np.abs(out_all - out_basic)))
print(f"ORT_ENABLE_ALL  [0,0,:3]: {out_all[0,0,:3]}")
print(f"ORT_ENABLE_BASIC [0,0,:3]: {out_basic[0,0,:3]}")
print(f"Max diff fused vs unfused FP16: {max_diff:.4f}")
PASS = (max_diff < 1.0)
print(f"PASS={PASS}")
