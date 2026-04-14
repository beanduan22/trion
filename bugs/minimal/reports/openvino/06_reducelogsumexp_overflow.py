"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ BUG REPORT — OpenVINO CPU plugin: ReduceLogSumExp overflows for x ≥ 88.7     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Affected compiler : OpenVINO 2026.0 CPU plugin
Related GitHub    : openvinotoolkit/openvino#32839
Severity          : High (silent +inf output for common input ranges)
Date              : 2026-04-14

──────────────────────────────────────────────────────────────────────────────
WHAT IS WRONG
──────────────────────────────────────────────────────────────────────────────
ReduceLogSumExp(x) := log(Σᵢ exp(xᵢ)).  For any x ≥ 88.7 (fp32 exp overflow
boundary), exp(x) overflows to +inf in fp32 → sum is +inf → log is +inf.
OV implements the naive formula, so any reasonably-large input produces +inf.
Correct implementations use the max-shift trick:
    m = max(x)   →   result = m + log(Σ exp(xᵢ - m))
since all `xᵢ - m ≤ 0`, `exp(·)` cannot overflow.

──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE
──────────────────────────────────────────────────────────────────────────────
OV's `ReduceLogSumExp` kernel is implemented as three sequential ops:
  1. Exp(x)     ← overflows for x ≥ 88.7
  2. ReduceSum  ← propagates +inf
  3. Log        ← log(+inf) = +inf
The max-subtraction stabilizer is never applied.

──────────────────────────────────────────────────────────────────────────────
CROSS-COMPILER VERIFICATION (2026-04-14)
──────────────────────────────────────────────────────────────────────────────
Input: x = [[100.0, 88.0, 50.0], [200.0, -10.0, 1.0]]
Stable reference: [100.00001, 200.0]

  numpy (stable) → [100.00001, 200.0]  ✓
  ONNX-Reference → [100.00001, 200.0]  ✓
  ORT            → [100.00001, 200.0]  ✓
  PyTorch eager  → [100.00001, 200.0]  ✓
  Inductor       → [100.00001, 200.0]  ✓
  TensorFlow     → [100.00001, 200.0]  ✓
  JAX            → [100.00001, 200.0]  ✓
  onnx2torch     → [100.00001, 200.0]  ✓
  OpenVINO CPU   → [   +inf,    +inf]  ✗ OVERFLOW

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

x = np.array([[100.0, 88.0, 50.0],
              [200.0, -10.0, 1.0]], dtype=np.float32)

# Stable reference (max-shift trick)
xd = x.astype(np.float64)
m_ = xd.max(axis=1, keepdims=True)
expected = (m_ + np.log(np.exp(xd - m_).sum(axis=1, keepdims=True))).squeeze(1).astype(np.float32)

# Build minimal ReduceLogSumExp model
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node  = helper.make_node("ReduceLogSumExp", ["X"], ["Y"], axes=[1], keepdims=0)
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out = m({"X": x})[m.output(0)]

print(f"Input      =\n{x}")
print(f"Expected   = {expected}   (stable formula)")
print(f"OV output  = {out}")

has_inf = bool(np.any(np.isinf(out)))
if not has_inf and np.max(np.abs(out - expected)) < 0.1:
    print("\n✓ OK"); sys.exit(1)
print("\n✗ BUG — OV ReduceLogSumExp overflows on common input ranges")
print("         Fix: apply max-shift stabilizer before exp().")
sys.exit(0)
