"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: int8 Add saturates instead of wrapping     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : extends openvinotoolkit/openvino#33518 to signed int8
Severity          : High (silent wrong results; spec violation)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
ONNX int8 Add: two's-complement wrap. 100 + 100 = 200 → should wrap to -56.
OV CPU returns 127 (saturates).

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
Signed int8 Add path uses `_mm_adds_epi8` (saturating) instead of `_mm_add_epi8`.
Completes the 6-op family of 8-bit saturation bugs in OV CPU:
uint8 {Add, Sub, Mul} and int8 {Add, Sub} — all share the same root cause
(wrong SIMD intrinsic family).

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: a=[100, 127, -128, -100], b=[100, 10, -10, -50]
a+b (int): [200, 137, -138, -150]
Expected (wrap): [-56, -119, 118, 106]

  numpy / ONNX-Ref / PyTorch / TF / JAX / onnx2torch → [-56,-119, 118, 106] ✓
  OpenVINO CPU → [127, 127, -128, -128]  ✗ saturates to bounds

Exit 0 = bug reproduces   Exit 1 = does not reproduce   Exit 2 = deps missing
"""
import sys
import numpy as np

try:
    import openvino as ov
    import onnx
    from onnx import TensorProto, helper
except ImportError as e:
    print(f"SKIP: {e}"); sys.exit(2)

print(f"OpenVINO version: {ov.__version__}")

a = np.array([ 100,  127, -128, -100], dtype=np.int8)
b = np.array([ 100,   10,  -10,  -50], dtype=np.int8)

A = helper.make_tensor_value_info("A", TensorProto.INT8, [4])
B = helper.make_tensor_value_info("B", TensorProto.INT8, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.INT8, None)
node  = helper.make_node("Add", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [A, B], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"A": a, "B": b})[m.output(0)]

expected = (a.astype(np.int16) + b.astype(np.int16)).astype(np.int8)

print(f"a          = {a}")
print(f"b          = {b}")
print(f"a+b (i16)  = {a.astype(np.int16) + b.astype(np.int16)}")
print(f"Expected   = {expected}   (two's complement)")
print(f"OV output  = {out}")

if np.array_equal(out, expected):
    print("\n✓ OK"); sys.exit(1)
print("\n✗ BUG — OV int8 Add saturates")
sys.exit(0)
