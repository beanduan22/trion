#!/usr/bin/env python3
"""
Bug ID     : ort_ada_layer_norm
Source     : Campaign v1 (fuzzing)
Compiler   : OnnxRuntime (ORT_ENABLE_ALL)
Patterns   : ada_layer_norm
Root cause : ORT fuses the adaptive LayerNorm (SkipLayerNorm pattern) with incorrect parameter mapping, mixing scale and bias
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

# Shape: [1, 8, 16] — batch=1, seq_len=8, hidden=16
BATCH, SEQ, HIDDEN = 1, 8, 16

# Adaptive LayerNorm parameters (per-channel)
ln_scale = np.ones(HIDDEN, dtype=np.float32)       # LayerNorm scale (gamma)
ln_bias  = np.zeros(HIDDEN, dtype=np.float32)      # LayerNorm bias (beta)
ada_scale = np.random.randn(HIDDEN).astype(np.float32)  # adaptive scale from another input
ada_shift = np.random.randn(HIDDEN).astype(np.float32)  # adaptive shift from another input

ln_scale_init  = onh.from_array(ln_scale,  name="ln_scale")
ln_bias_init   = onh.from_array(ln_bias,   name="ln_bias")
ada_scale_init = onh.from_array(ada_scale, name="ada_scale")
ada_shift_init = onh.from_array(ada_shift, name="ada_shift")

# Graph: X + Skip -> LayerNorm(ln_scale, ln_bias) -> Mul(ada_scale) -> Add(ada_shift) -> Output
skip = np.zeros((BATCH, SEQ, HIDDEN), dtype=np.float32)  # zero skip for clarity
skip_init = onh.from_array(skip, name="skip")

add_skip_node = oh.make_node("Add", inputs=["input", "skip"], outputs=["x_skip"])

ln_node = oh.make_node(
    "LayerNormalization",
    inputs=["x_skip", "ln_scale", "ln_bias"],
    outputs=["ln_out"],
    axis=-1,
    epsilon=1e-5,
)

mul_node = oh.make_node("Mul", inputs=["ln_out", "ada_scale"], outputs=["scaled"])
add_node = oh.make_node("Add", inputs=["scaled", "ada_shift"], outputs=["output"])

input_t  = oh.make_tensor_value_info("input",  onnx.TensorProto.FLOAT, [BATCH, SEQ, HIDDEN])
output_t = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [BATCH, SEQ, HIDDEN])

graph = oh.make_graph(
    [add_skip_node, ln_node, mul_node, add_node],
    "ada_layer_norm",
    [input_t],
    [output_t],
    initializer=[skip_init, ln_scale_init, ln_bias_init, ada_scale_init, ada_shift_init],
)

model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)

x = np.random.randn(BATCH, SEQ, HIDDEN).astype(np.float32)

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
