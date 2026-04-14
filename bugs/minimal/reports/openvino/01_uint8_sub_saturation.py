"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: uint8 Sub saturates instead of wrapping    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : openvinotoolkit/openvino#33518
Severity          : High (silent wrong results; spec violation)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
ONNX elem_type 2 (uint8) arithmetic MUST use modular (wrapping) semantics —
exactly like NumPy/C unsigned types: `(a - b) mod 256`.
OpenVINO CPU uses SATURATING semantics instead: any underflow is clamped to 0.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
OV's CPU `ElementwiseBroadcast` kernel dispatches to `_mm_subs_epu8` (the
SATURATING x86 SIMD intrinsic) for the uint8 path.  The non-saturating
equivalent `_mm_sub_epi8` exists and must be used to match ONNX semantics.
Cross-check confirms 16/32/64-bit integer paths wrap correctly — only 8-bit
is affected.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: a=[5, 200, 250, 0], b=[10, 100, 10, 50]
Expected (mod 256): [251, 100, 240, 206]

  numpy          → [251, 100, 240, 206]  ✓
  ONNX-Reference → [251, 100, 240, 206]  ✓
  onnx2torch     → [251, 100, 240, 206]  ✓
  PyTorch eager  → [251, 100, 240, 206]  ✓
  TensorFlow     → [251, 100, 240, 206]  ✓
  JAX            → [251, 100, 240, 206]  ✓
  OpenVINO CPU   → [  0, 100, 240,   0]  ✗ SATURATES

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

a = np.array([  5, 200, 250,   0], dtype=np.uint8)
b = np.array([ 10, 100,  10,  50], dtype=np.uint8)

# Build minimal Sub model
A = helper.make_tensor_value_info("A", TensorProto.UINT8, [4])
B = helper.make_tensor_value_info("B", TensorProto.UINT8, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.UINT8, None)
node  = helper.make_node("Sub", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [A, B], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# Run OV CPU
core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"A": a, "B": b})[m.output(0)]

# Expected: modular arithmetic
expected = ((a.astype(np.int16) - b.astype(np.int16)) % 256).astype(np.uint8)

print(f"a          = {a}")
print(f"b          = {b}")
print(f"Expected   = {expected}   (mod 256, per ONNX spec)")
print(f"OV output  = {out}")

if np.array_equal(out, expected):
    print("\n✓ OK — OV matches ONNX spec")
    sys.exit(1)
print("\n✗ BUG — OV saturates (5-10=0, should be 251)")
sys.exit(0)
