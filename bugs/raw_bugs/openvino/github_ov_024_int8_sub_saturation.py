#!/usr/bin/env python3
"""
Bug ID     : github_ov_024_int8_sub_saturation
Source     : Independent discovery 2026-04-14 (extends OV #33518 to int8)
Compiler   : OpenVINO 2026.0 CPU
Patterns   : int8 Sub arithmetic overflow
Root cause : Same root cause as github_ov_019–021: OpenVINO CPU plugin uses saturating
             SIMD instructions for all integer element-wise ops, regardless of dtype.
             For int8 Sub, overflow should produce two's-complement wrap:
               -128 - 1 = -129 → wraps to  127 (add 256)
               127 - (-128) = 255 → wraps to  -1 (subtract 256)
             OV saturates at [-128, 127] instead, giving -128 and 127 respectively.
             Only the non-overflowing cases (-1-(-127)=126, 0-100=-100) are correct.

             Observed (a=[-128, 127, -1, 0], b=[1, -128, -127, 100]):
               wrap_ref : [ 127,  -1,  126, -100]  ← ONNX two's complement
               sat_ref  : [-128, 127,  126, -100]  ← saturating
               OV out   : [-128, 127,  126, -100]  ← matches sat, not wrap → BUG

Tolerance  : exact integer equality

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import sys

try:
    import openvino as ov
except ImportError:
    print("SKIP: openvino not installed")
    sys.exit(2)

a = np.array([-128,  127,  -1,   0], dtype=np.int8)
b = np.array([   1, -128, -127, 100], dtype=np.int8)

# ONNX spec: two's complement wrapping (same as C uint8 cast of int16 result)
ref_wrap = (a.astype(np.int16) - b.astype(np.int16)).astype(np.int8)
# Saturation (wrong)
ref_sat  = np.clip(a.astype(np.int16) - b.astype(np.int16), -128, 127).astype(np.int8)

A_vi = helper.make_tensor_value_info("A", TensorProto.INT8, [4])
B_vi = helper.make_tensor_value_info("B", TensorProto.INT8, [4])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.INT8, None)
node = helper.make_node("Sub", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [A_vi, B_vi], [Y_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
try:
    ov_m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
    out = ov_m({"A": a, "B": b})[ov_m.output(0)]
except Exception as e:
    print(f"OpenVINO error: {e}")
    sys.exit(2)

print(f"a        : {a}")
print(f"b        : {b}")
print(f"OV out   : {out}")
print(f"wrap ref : {ref_wrap}  (correct — two's complement)")
print(f"sat  ref : {ref_sat}   (wrong  — saturation)")

is_sat  = np.array_equal(out, ref_sat)
is_wrap = np.array_equal(out, ref_wrap)

print(f"\nMatches wrap (correct)? {is_wrap}")
print(f"Matches sat  (wrong)?   {is_sat}")

PASS = is_sat and not is_wrap
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — OV int8 Sub saturates instead of two's-complement wrapping")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
