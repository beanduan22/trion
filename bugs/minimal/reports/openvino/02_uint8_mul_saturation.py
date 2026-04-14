"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: uint8 Mul saturates instead of wrapping    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : openvinotoolkit/openvino#33518 (Mul variant)
Severity          : High (silent wrong results; spec violation)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
ONNX uint8 Mul must produce `(a * b) mod 256`.  OV CPU clamps the product
to 255 whenever `a * b > 255`.  Example: 200 * 200 = 40000 → should wrap to
40000 mod 256 = 64; OV returns 255.  Applies to all uint8 Mul results ≥ 256.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
Same dispatch bug as the Sub variant: OV's CPU uint8 Mul kernel uses a
saturating implementation.  uint8 * uint8 requires an int16 intermediate and
a truncating cast back to uint8; OV is instead using signed-saturating SIMD
(or a manual clamp) that clips to [0, 255].

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: a=[200, 100, 50, 10], b=[200, 3, 6, 30]
a*b    = [40000, 300, 300, 300]
Expected (mod 256): [64, 44, 44, 44]

  numpy          → [64, 44, 44, 44]  ✓
  ONNX-Reference → [64, 44, 44, 44]  ✓
  onnx2torch     → [64, 44, 44, 44]  ✓
  PyTorch eager  → [64, 44, 44, 44]  ✓
  TensorFlow     → [64, 44, 44, 44]  ✓
  JAX            → [64, 44, 44, 44]  ✓
  OpenVINO CPU   → [255, 255, 255, 255]  ✗ SATURATES to 255

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

a = np.array([200, 100,  50,  10], dtype=np.uint8)
b = np.array([200,   3,   6,  30], dtype=np.uint8)

A = helper.make_tensor_value_info("A", TensorProto.UINT8, [4])
B = helper.make_tensor_value_info("B", TensorProto.UINT8, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.UINT8, None)
node  = helper.make_node("Mul", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [A, B], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"A": a, "B": b})[m.output(0)]

expected = ((a.astype(np.int32) * b.astype(np.int32)) % 256).astype(np.uint8)

print(f"a          = {a}")
print(f"b          = {b}")
print(f"a*b (i32)  = {a.astype(np.int32) * b.astype(np.int32)}")
print(f"Expected   = {expected}   (mod 256, per ONNX spec)")
print(f"OV output  = {out}")

if np.array_equal(out, expected):
    print("\n✓ OK — OV matches ONNX spec")
    sys.exit(1)
print("\n✗ BUG — OV saturates to 255 (expected wrap mod 256)")
sys.exit(0)
