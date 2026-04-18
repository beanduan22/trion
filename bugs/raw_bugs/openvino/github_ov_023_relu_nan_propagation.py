#!/usr/bin/env python3
"""
Bug ID     : github_ov_023_relu_nan_propagation
Source     : Independent discovery 2026-04-14 (cross-compiler differential)
Compiler   : OpenVINO 2026.0 CPU
Patterns   : Relu NaN propagation IEEE 754
Root cause : OpenVINO CPU Relu returns 0.0 for NaN inputs instead of propagating NaN.
             ONNX Relu spec: output = max(0, x).  Per IEEE 754, max(0, NaN) = NaN
             (quiet NaN propagates through comparisons).  ORT, PyTorch, and JAX all
             return NaN for Relu(NaN).  OpenVINO returns 0.0.
             This means OV silently converts NaN sentinels to 0 in any computation
             that uses Relu after a NaN-producing op (e.g. 0/0, sqrt(-1), inf-inf).
             Models relying on NaN propagation for padding masks or validity tracking
             will silently compute wrong values on OV CPU.

             Observed:
               input : [NaN, -1.0, 0.0, 1.0, inf]
               ORT   : [NaN,  0.0, 0.0, 1.0, inf]  ← NaN propagated (correct)
               OV    : [0.0,  0.0, 0.0, 1.0, inf]  ← NaN → 0.0 (wrong)

Tolerance  : exact NaN presence check

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import sys

try:
    import openvino as ov
except ImportError:
    print("SKIP: openvino not installed")
    sys.exit(2)

x = np.array([float('nan'), -1.0, 0.0, 1.0, float('inf')], dtype=np.float32)

# Build minimal Relu model
X_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("Relu", ["X"], ["Y"])
graph = helper.make_graph([node], "g", [X_vi], [Y_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 6)])

model_bytes = model.SerializeToString()
mb = model_bytes

# ORT reference (correct: NaN propagates)
sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

# OpenVINO output
core = ov.Core()
try:
    ov_m = core.compile_model(core.read_model(model_bytes, b""), "CPU")
    ov_out = ov_m({"X": x})[ov_m.output(0)]
except Exception as e:
    print(f"OpenVINO error: {e}")
    sys.exit(2)

print(f"Input  : {x}")
print(f"ORT    : {ort_out}  ← NaN propagated (correct)")
print(f"OV     : {ov_out}")

# The bug: OV outputs 0.0 where NaN is expected
ort_nan = bool(np.isnan(ort_out[0]))
ov_nan  = bool(np.isnan(ov_out[0]))

print(f"\nInput[0] is NaN: True")
print(f"ORT[0]  is NaN: {ort_nan}  (expected: True)")
print(f"OV[0]   is NaN: {ov_nan}  (expected: True)")

# Bug: OV converts NaN to 0
PASS = (not ov_nan) and ort_nan
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754)")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
