"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: Exp(NaN) returns +inf instead of NaN       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : new discovery (no existing issue found)
Severity          : Medium-High (IEEE 754 violation; different failure mode
                    than Relu — NaN becomes +inf, which then amplifies in any
                    downstream op using exp's output)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
IEEE 754 mandates `exp(NaN) = NaN`.  OV CPU returns +∞.
Unlike Relu (which returns 0 for NaN), Exp converts NaN to +inf.  Both
exhibit the same class of bug — sanitizing input domain with ordered
comparisons that silently strip NaN — but produce different garbage.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
OV's Exp kernel uses range-based short-circuiting:
  - if x > LARGE_POS_THRESHOLD  →  return +inf  (avoid computing exp)
  - if x < LARGE_NEG_THRESHOLD  →  return 0     (avoid computing exp)
  - else                        →  compute exp(x)
When x is NaN, the ordered comparison `x > LARGE_POS_THRESHOLD` returns false
and `x < LARGE_NEG_THRESHOLD` returns false too.  But depending on how the
compiler emits the branch (e.g., a `cmpltps` + bitmask), NaN's unordered
result can map into either branch.  The observed failure is "NaN → +inf"
(the large-positive branch), which is consistent with a signed-compare that
treats NaN's bit pattern as > 0.

Correct implementations use the CPU's native `expf`/`vexpf` intrinsics which
preserve NaN automatically, or do an explicit NaN check before the range
short-circuit.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: x = [NaN, 1.0]
Expected: [NaN, 2.71828]

  ONNX-Reference → [NaN, 2.71828]  ✓
  ORT            → [NaN, 2.71828]  ✓
  PyTorch eager  → [NaN, 2.71828]  ✓
  Inductor       → [NaN, 2.71828]  ✓
  TF eager       → [NaN, 2.71828]  ✓
  TF-XLA         → [NaN, 2.71828]  ✓
  JAX            → [NaN, 2.71828]  ✓
  JAX-jit        → [NaN, 2.71828]  ✓
  onnx2torch     → [NaN, 2.71828]  ✓
  OpenVINO CPU   → [+inf, 2.71828] ✗ NaN→+inf

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

x = np.array([float('nan'), 1.0], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node  = helper.make_node("Exp", ["X"], ["Y"])
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"X": x})[m.output(0)]

print(f"Input      = {x}")
print(f"Expected   = [NaN, 2.71828]  (IEEE 754: exp(NaN) = NaN)")
print(f"OV output  = {out}")

ov_nan = bool(np.isnan(out[0]))
ov_inf = bool(np.isinf(out[0]))
if ov_nan:
    print("\n✓ OK — NaN propagated"); sys.exit(1)
print(f"\n✗ BUG — OV Exp(NaN) = {out[0]} ({'+inf' if ov_inf else 'other'}); should be NaN")
print("         Fix: add explicit NaN check before range short-circuit.")
sys.exit(0)
