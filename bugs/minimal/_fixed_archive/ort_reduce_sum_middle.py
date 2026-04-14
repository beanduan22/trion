#!/usr/bin/env python3
"""
Bug ID     : ort_reduce_sum_middle
Source     : Campaign v1 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : reduce_sum_middle_axis
Root cause : ORT optimizer incorrectly transposes axes during a reduce fusion, summing along the wrong dimension
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""

import sys as _sys

try:
    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

np.random.seed(42)

# Input [1, 4, 8, 8]
# ReduceSum(axis=1, keepdims=1) -> [1, 1, 8, 8]
# Reshape -> [1, 64]
# MatMul weight [64, 8] -> [1, 8]

INPUT_SHAPE   = [1, 4, 8, 8]
REDUCED_SHAPE = [1, 1, 8, 8]
FLAT_DIM      = 64
OUT_DIM       = 8

weight = np.random.randn(FLAT_DIM, OUT_DIM).astype(np.float32)
weight_init = onh.from_array(weight, name="weight")

axes_init = onh.from_array(np.array([1], dtype=np.int64), name="axes")
reshape_shape_init = onh.from_array(
    np.array([1, FLAT_DIM], dtype=np.int64), name="reshape_shape"
)

reduce_node = oh.make_node(
    "ReduceSum",
    inputs=["input", "axes"],
    outputs=["reduced"],
    keepdims=1,
)

reshape_node = oh.make_node(
    "Reshape",
    inputs=["reduced", "reshape_shape"],
    outputs=["flat"],
)

matmul_node = oh.make_node(
    "MatMul",
    inputs=["flat", "weight"],
    outputs=["output"],
)

input_t  = oh.make_tensor_value_info("input",  onnx.TensorProto.FLOAT, INPUT_SHAPE)
output_t = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, OUT_DIM])

graph = oh.make_graph(
    [reduce_node, reshape_node, matmul_node],
    "reduce_sum_middle",
    [input_t],
    [output_t],
    initializer=[axes_init, reshape_shape_init, weight_init],
)

model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 7
onnx.checker.check_model(model)

x = np.random.randn(*INPUT_SHAPE).astype(np.float32)

# Reference: ORT_DISABLE_ALL
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model.SerializeToString(), sess_options=so_ref,
                                 providers=["CPUExecutionProvider"])
ref_out = sess_ref.run(["output"], {"input": x})[0]

# Optimized: ORT_ENABLE_ALL
so_opt = ort.SessionOptions()
so_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opt = ort.InferenceSession(model.SerializeToString(), sess_options=so_opt,
                                 providers=["CPUExecutionProvider"])
opt_out = sess_opt.run(["output"], {"input": x})[0]

diff = float(np.max(np.abs(opt_out - ref_out)))
TOL = 0.01
PASS = diff <= TOL

print(f"max_diff={diff:.6f}  tol={TOL}  pass={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
