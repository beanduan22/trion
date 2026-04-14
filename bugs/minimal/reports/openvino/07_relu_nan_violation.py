"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: Relu(NaN) returns 0.0 instead of NaN       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : new discovery (no existing issue found)
Severity          : Medium-High (IEEE 754 violation; silent NaN loss)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
Per ONNX spec: `Relu(x) = max(0, x)`.
Per IEEE 754: `max(0, NaN) = NaN` — NaN propagates through comparisons.
OV returns 0.0 for Relu(NaN), silently discarding the NaN sentinel.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
OV's Relu kernel likely uses `x > 0 ? x : 0` with an ordered comparison.
For IEEE 754 ordered comparisons, any comparison involving NaN returns false.
So NaN > 0  →  false  →  falls into the `else 0.0` branch.
Correct implementations use `fmaxf(0, x)` (which handles NaN per IEEE 754)
or an unordered comparison with explicit NaN passthrough.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: x = [NaN, -1.0, 0.0, 1.0]
Expected: [NaN, 0.0, 0.0, 1.0]

  ONNX-Reference → [NaN, 0.0, 0.0, 1.0]  ✓
  ORT            → [NaN, 0.0, 0.0, 1.0]  ✓
  PyTorch eager  → [NaN, 0.0, 0.0, 1.0]  ✓
  Inductor       → [NaN, 0.0, 0.0, 1.0]  ✓
  TF eager       → [NaN, 0.0, 0.0, 1.0]  ✓
  TF-XLA         → [NaN, 0.0, 0.0, 1.0]  ✓
  JAX            → [NaN, 0.0, 0.0, 1.0]  ✓
  JAX-jit        → [NaN, 0.0, 0.0, 1.0]  ✓
  onnx2torch     → [NaN, 0.0, 0.0, 1.0]  ✓
  OpenVINO CPU   → [0.0, 0.0, 0.0, 1.0]  ✗ NaN silenced

Sibling activations in OV 2026.0: Sigmoid / Tanh / Sqrt / Abs / Neg / Log all
propagate NaN correctly — only Relu (this bug) and Exp (see 08_exp_nan_*) are
affected.

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

x = np.array([float('nan'), -1.0, 0.0, 1.0], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node  = helper.make_node("Relu", ["X"], ["Y"])
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"X": x})[m.output(0)]

print(f"Input      = {x}")
print(f"Expected   = [NaN, 0, 0, 1]  (IEEE 754: max(0, NaN) = NaN)")
print(f"OV output  = {out}")

ov_nan = bool(np.isnan(out[0]))
if ov_nan:
    print("\n✓ OK — NaN propagated"); sys.exit(1)
print(f"\n✗ BUG — OV Relu(NaN) = {out[0]} (should be NaN)")
print("         Fix: use fmaxf(0, x) or explicit NaN check before comparison.")
sys.exit(0)
