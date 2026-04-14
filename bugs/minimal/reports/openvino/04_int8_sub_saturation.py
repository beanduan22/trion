"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: int8 Sub saturates instead of wrapping     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : extends openvinotoolkit/openvino#33518 to signed int8
Severity          : High (silent wrong results; spec violation)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
ONNX signed int8 Sub must use two's-complement wrapping, range [-128, 127].
OV saturates instead: -128 - 1 should be 127 (wrap), OV returns -128.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
Signed int8 path uses `_mm_subs_epi8` (saturating signed SIMD) instead of
`_mm_sub_epi8` (wrapping).  Note: OV int16/int32/int64 Sub paths all wrap
correctly — the bug is confined to the 8-bit SIMD dispatch.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: a=[-128, 127, -1, 0], b=[1, -128, -127, 100]
Expected (two's complement wrap): [127, -1, 126, -100]

  numpy / ONNX-Ref / PyTorch / TF / JAX / onnx2torch → [127, -1, 126, -100] ✓
  OpenVINO CPU → [-128, 127, 126, -100]  ✗ saturates boundary cases

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

a = np.array([-128,  127,   -1,    0], dtype=np.int8)
b = np.array([   1, -128, -127,  100], dtype=np.int8)

A = helper.make_tensor_value_info("A", TensorProto.INT8, [4])
B = helper.make_tensor_value_info("B", TensorProto.INT8, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.INT8, None)
node  = helper.make_node("Sub", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [A, B], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"A": a, "B": b})[m.output(0)]

# Two's complement wrap: int16 subtract, truncate to int8
expected = (a.astype(np.int16) - b.astype(np.int16)).astype(np.int8)

print(f"a          = {a}")
print(f"b          = {b}")
print(f"Expected   = {expected}   (two's complement)")
print(f"OV output  = {out}")

if np.array_equal(out, expected):
    print("\n✓ OK"); sys.exit(1)
print("\n✗ BUG — OV int8 Sub saturates at [-128, 127]")
sys.exit(0)
