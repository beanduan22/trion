#!/usr/bin/env python3
"""
Bug ID     : ort_relu_add_relu
Source     : Campaign v1 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : relu_add_relu
Root cause : ORT fuses the double-ReLU with residual into a single activation, incorrectly handling the intermediate Add
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

# Shape: [1, 4, 8, 8]
INPUT_SHAPE = [1, 4, 8, 8]

# Y is a negative-valued residual initializer to stress the double-ReLU pattern
y_data = -np.abs(np.random.randn(*INPUT_SHAPE).astype(np.float32)) - 0.5
y_init = onh.from_array(y_data, name="y")

# Graph: X -> ReLU -> Add(Y) -> ReLU -> Output
relu1_node = oh.make_node("Relu",  inputs=["input"],  outputs=["relu1_out"])
add_node   = oh.make_node("Add",   inputs=["relu1_out", "y"], outputs=["add_out"])
relu2_node = oh.make_node("Relu",  inputs=["add_out"], outputs=["output"])

input_t  = oh.make_tensor_value_info("input",  onnx.TensorProto.FLOAT, INPUT_SHAPE)
output_t = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, INPUT_SHAPE)

graph = oh.make_graph(
    [relu1_node, add_node, relu2_node],
    "relu_add_relu",
    [input_t],
    [output_t],
    initializer=[y_init],
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
