"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: uint8 Add saturates instead of wrapping    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : openvinotoolkit/openvino#33518 (Add variant)
Severity          : High (silent wrong results; spec violation)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
ONNX uint8 Add: `(a + b) mod 256`.  OV CPU clamps overflow to 255.
Example: 200 + 100 = 300 → expected 44, OV returns 255.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
OV uint8 Add kernel uses `_mm_adds_epu8` (unsigned saturating SIMD) instead
of `_mm_add_epi8` (wrapping).  Same class of bug as Sub/Mul — the 8-bit
element-wise path uses the wrong intrinsic family.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: a=[200, 100, 255, 128], b=[100, 200, 10, 200]
a+b (int): [300, 300, 265, 328]
Expected (mod 256): [44, 44, 9, 72]

  numpy / ONNX-Ref / PyTorch / TF / JAX / onnx2torch → [44, 44, 9, 72] ✓
  OpenVINO CPU → [255, 255, 255, 255]  ✗ saturates

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

a = np.array([200, 100, 255, 128], dtype=np.uint8)
b = np.array([100, 200,  10, 200], dtype=np.uint8)

A = helper.make_tensor_value_info("A", TensorProto.UINT8, [4])
B = helper.make_tensor_value_info("B", TensorProto.UINT8, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.UINT8, None)
node  = helper.make_node("Add", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [A, B], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"A": a, "B": b})[m.output(0)]

expected = ((a.astype(np.int32) + b.astype(np.int32)) % 256).astype(np.uint8)

print(f"a          = {a}")
print(f"b          = {b}")
print(f"a+b (i32)  = {a.astype(np.int32) + b.astype(np.int32)}")
print(f"Expected   = {expected}   (mod 256)")
print(f"OV output  = {out}")

if np.array_equal(out, expected):
    print("\n✓ OK"); sys.exit(1)
print("\n✗ BUG — OV uint8 Add saturates to 255")
sys.exit(0)
