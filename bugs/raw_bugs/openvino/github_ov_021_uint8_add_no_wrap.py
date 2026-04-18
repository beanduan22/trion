#!/usr/bin/env python3
"""
Bug ID     : github_ov_021_uint8_add_no_wrap
Source     : GitHub — openvinotoolkit/openvino#33518 (Add variant) + 2026-04-14
Compiler   : OpenVINO 2026.0 CPU
Patterns   : uint8 Add arithmetic overflow
Root cause : Same root cause as github_ov_019/020: OV CPU plugin applies saturation
             to uint8 Add.  ONNX spec mandates wrapping: (200+100) mod 256 = 44.
             OV returns 255.  The bug affects all three arithmetic ops (Add/Sub/Mul)
             on uint8 tensors, indicating the underlying CPU kernel dispatch path
             for UINT8 elements uses a saturating SIMD instruction throughout.
             Spec reference: ONNX elem_type 2 (UINT8) arithmetic — wrapping semantics
             (same as C unsigned arithmetic, same as numpy uint8 arithmetic).

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

# Overflow cases: a + b > 255 → must wrap
a = np.array([200, 100, 255, 128], dtype=np.uint8)
b = np.array([100, 200,  10, 200], dtype=np.uint8)

ref_wrap = ((a.astype(np.int32) + b.astype(np.int32)) % 256).astype(np.uint8)
ref_sat  = np.clip(a.astype(np.int32) + b.astype(np.int32), 0, 255).astype(np.uint8)

A_vi = helper.make_tensor_value_info("A", TensorProto.UINT8, [4])
B_vi = helper.make_tensor_value_info("B", TensorProto.UINT8, [4])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.UINT8, None)
node = helper.make_node("Add", ["A", "B"], ["Y"])
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
print(f"a+b      : {a.astype(np.int32)+b.astype(np.int32)}")
print(f"OV out   : {out}")
print(f"wrap ref : {ref_wrap}  (correct)")
print(f"sat  ref : {ref_sat}   (wrong)")

is_sat  = np.array_equal(out, ref_sat)
is_wrap = np.array_equal(out, ref_wrap)

PASS = is_sat and not is_wrap
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — OV uint8 Add saturates instead of wrapping")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
