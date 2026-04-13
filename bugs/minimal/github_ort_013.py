#!/usr/bin/env python3
"""
Bug ID     : github_ort_013
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : instance norm fp16 wrong
Root cause : Bug: ORT GPU InstanceNorm FP16 accumulated in FP16, wrong for large-variance inputs (PR#9879).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Bug: ORT GPU InstanceNorm FP16 accumulated in FP16, wrong for large-variance inputs (PR#9879).
np.random.seed(41)
x_fp32 = (np.random.randn(1, 4, 8, 8) * 50).astype(np.float32)
x_fp16 = x_fp32.astype(np.float16)

# FP32 model
X32 = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8, 8])
Y32 = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8, 8])
sc32 = numpy_helper.from_array(np.ones(4, dtype=np.float32),  "scale")
bi32 = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "bias")
n32  = helper.make_node("InstanceNormalization", ["X","scale","bias"], ["Y"], epsilon=1e-5)
g32  = helper.make_graph([n32], "g", [X32], [Y32], initializer=[sc32, bi32])
m32  = helper.make_model(g32, opset_imports=[helper.make_opsetid("", 6)])
sess32 = ort.InferenceSession(m32.SerializeToString(), providers=["CPUExecutionProvider"])
out32  = sess32.run(None, {"X": x_fp32})[0]

# FP16 model
X16 = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 4, 8, 8])
Y16 = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 4, 8, 8])
sc16 = numpy_helper.from_array(np.ones(4, dtype=np.float16),  "scale")
bi16 = numpy_helper.from_array(np.zeros(4, dtype=np.float16), "bias")
n16  = helper.make_node("InstanceNormalization", ["X","scale","bias"], ["Y"], epsilon=1e-5)
g16  = helper.make_graph([n16], "g", [X16], [Y16], initializer=[sc16, bi16])
m16  = helper.make_model(g16, opset_imports=[helper.make_opsetid("", 6)])
sess16 = ort.InferenceSession(m16.SerializeToString(), providers=["CPUExecutionProvider"])
out16  = sess16.run(None, {"X": x_fp16})[0].astype(np.float32)

max_diff = float(np.max(np.abs(out32 - out16)))
print(f"FP32 output[0,0,0,:4]: {out32[0,0,0,:4]}")
print(f"FP16 output[0,0,0,:4]: {out16[0,0,0,:4]}")
print(f"Max abs diff FP32 vs FP16: {max_diff:.6f}")
PASS = max_diff < 0.1
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
