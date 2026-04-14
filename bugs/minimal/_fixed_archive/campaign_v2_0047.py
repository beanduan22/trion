#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0047
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime
Patterns   : x^2 -> ReduceSum -> Add(eps) -> Sqrt -> Div  (L2 norm) then Div+Mul+Add chain
Root cause : ORT may fuse the L2-norm pattern into an LpNormalization op, then the
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(6)

N, D = 1, 16

def make_model():
    nodes = [
        # L2 norm: x^2 -> ReduceSum(last) -> Add(eps) -> Sqrt -> Div
        helper.make_node("Mul", ["x", "x"], ["sq"]),
        helper.make_node("ReduceSum", ["sq", "l2_ax"], ["l2_sum"], keepdims=0),
        helper.make_node("Add", ["l2_sum", "l2_eps"], ["l2_add"]),
        helper.make_node("Sqrt", ["l2_add"], ["l2_sqrt"]),
        helper.make_node("Div", ["x", "l2_sqrt"], ["l2_norm"]),
        # DBC: double blend with constants
        helper.make_node("Div", ["l2_norm", "dbc_c"], ["dbc_div"]),
        helper.make_node("Mul", ["l2_norm", "dbc_ic"], ["dbc_mul"]),
        helper.make_node("Add", ["dbc_div", "dbc_mul"], ["dbc_sum"]),
        helper.make_node("Div", ["dbc_sum", "dbc_two"], ["dbc_out"]),
        # MAM: multiply-add-multiply chain
        helper.make_node("Mul", ["dbc_out", "mam_a"], ["mam_x1"]),
        helper.make_node("Add", ["mam_x1", "mam_b"], ["mam_x2"]),
        helper.make_node("Mul", ["mam_x2", "mam_c"], ["mam_out"]),
        # Min clamp
        helper.make_node("Min", ["mam_out", "clamp_hi"], ["out"]),
    ]

    initializers = [
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "l2_ax"),
        numpy_helper.from_array(np.array([1e-8], dtype=np.float32), "l2_eps"),
        numpy_helper.from_array(np.array([5.25], dtype=np.float32), "dbc_c"),
        numpy_helper.from_array(np.array([0.1904762], dtype=np.float32), "dbc_ic"),
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "dbc_two"),
        numpy_helper.from_array(np.array([1.2], dtype=np.float32), "mam_a"),
        numpy_helper.from_array(np.array([0.3], dtype=np.float32), "mam_b"),
        numpy_helper.from_array(np.array([1.5], dtype=np.float32), "mam_c"),
        numpy_helper.from_array(np.full((1, D), 10.0, dtype=np.float32), "clamp_hi"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0047",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, D])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, D])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_in})[0]

# Reference numpy implementation
def numpy_ref(x):
    sq = x * x
    l2_sum = sq.sum(axis=-1, keepdims=False)
    l2_norm = x / np.sqrt(l2_sum + 1e-8)
    dbc = (l2_norm / 5.25 + l2_norm * 0.1904762) / 2.0
    mam = (dbc * 1.2 + 0.3) * 1.5
    return np.minimum(mam, 10.0)

x_in = np.random.randn(N, D).astype(np.float32) * 2.0

out_all  = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis  = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
out_ref  = numpy_ref(x_in)

max_diff_ort = float(np.max(np.abs(out_all - out_dis)))
max_diff_all = float(np.max(np.abs(out_all - out_ref)))
max_diff_dis = float(np.max(np.abs(out_dis - out_ref)))

print(f"uid=0047 L2-norm + mul+add chain")
print(f"  ORT_ENABLE_ALL  vs numpy_ref       max_diff={max_diff_all:.6f}")
print(f"  ORT_DISABLE_ALL vs numpy_ref       max_diff={max_diff_dis:.6f}")
print(f"  ORT_ENABLE_ALL  vs ORT_DISABLE_ALL max_diff={max_diff_ort:.6f}")
passed = max_diff_ort < 0.01 and max_diff_all < 0.01
print(f"PASS={passed}")

PASS = passed
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
