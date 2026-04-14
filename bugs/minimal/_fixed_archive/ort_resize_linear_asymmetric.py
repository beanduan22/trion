#!/usr/bin/env python3
"""
Bug ID     : ort_resize_linear_asymmetric
Source     : Campaign v1 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : resize_linear_asymmetric_matmul
Root cause : ORT optimizer fuses linear asymmetric resize with downstream MatMul, reordering coordinate computation and producing wrong interpolated values
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

# Input shape [1, 1, 4, 4]; after 2x resize: [1, 1, 8, 8]
# Then flatten spatial dims and matmul with weight [64, 8]
# to get output [1, 1, 8, 8] -> reshape to [1, 64] -> matmul -> [1, 8]

INPUT_SHAPE = [1, 1, 4, 4]
RESIZED_H, RESIZED_W = 8, 8
FLAT_DIM = 1 * RESIZED_H * RESIZED_W   # 64
OUT_DIM = 8

weight = np.random.randn(FLAT_DIM, OUT_DIM).astype(np.float32)

# Build ONNX graph:
# Input [1,1,4,4]
# -> Resize(linear, asymmetric, scales=[1,1,2,2]) -> [1,1,8,8]
# -> Reshape([1, 64])
# -> MatMul(weight [64, 8])
# -> Output [1, 8]

scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
weight_init = onh.from_array(weight, name="weight")
scales_init = onh.from_array(scales, name="scales")

# ROI input (empty for non-tf_crop_and_resize)
roi_init = onh.from_array(np.array([], dtype=np.float32), name="roi")

reshape_shape = np.array([1, FLAT_DIM], dtype=np.int64)
reshape_shape_init = onh.from_array(reshape_shape, name="reshape_shape")

resize_node = oh.make_node(
    "Resize",
    inputs=["input", "roi", "scales"],
    outputs=["resized"],
    coordinate_transformation_mode="asymmetric",
    mode="linear",
    nearest_mode="floor",
)

reshape_node = oh.make_node(
    "Reshape",
    inputs=["resized", "reshape_shape"],
    outputs=["flat"],
)

matmul_node = oh.make_node(
    "MatMul",
    inputs=["flat", "weight"],
    outputs=["output"],
)

input_tensor = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, INPUT_SHAPE)
output_tensor = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, OUT_DIM])

graph = oh.make_graph(
    [resize_node, reshape_node, matmul_node],
    "resize_matmul",
    [input_tensor],
    [output_tensor],
    initializer=[roi_init, scales_init, reshape_shape_init, weight_init],
)

model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 7
onnx.checker.check_model(model)

x = np.random.randn(*INPUT_SHAPE).astype(np.float32)

# Run with optimizations disabled (reference)
sess_ref = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions(),
)
sess_ref.get_session_options().graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref_out = sess_ref.run(["output"], {"input": x})[0]

# Run with all optimizations enabled
sess_opt = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CPUExecutionProvider"],
)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opt2 = ort.InferenceSession(
    model.SerializeToString(),
    sess_options=so,
    providers=["CPUExecutionProvider"],
)
opt_out = sess_opt2.run(["output"], {"input": x})[0]

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
