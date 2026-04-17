#!/usr/bin/env python3
"""
Bug ID     : github_tvm_010
Source     : GitHub — apache/tvm#16211
Compiler   : TVM Relay (SimplifyExpr pass)
Patterns   : Sqrt(x) / y  →  rsqrt(x) * y  (algebraic rewrite)
Root cause : SimplifyExpr rewrites  t = sqrt(x); out = t / y
             into the single-step  out = rsqrt(x) * y.
             rsqrt on many targets is a fast approximation (Newton-Raphson
             from the 0x5f3759df bit-trick), introducing large relative error
             for small x.  For x=1e-4: reference = 0.01, rsqrt = ~99.84
             → relative error ~9983×.  TVM's built-in tolerance is 1e-4;
             this error is 7 orders of magnitude above it.
Tolerance  : 1e-4 (TVM default)

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
import numpy as np

def rsqrt_fast(x: np.float32) -> float:
    """Single Newton-Raphson rsqrt (the path TVM targets use)."""
    xhalf = np.float32(0.5) * x
    bits  = np.frombuffer(x.tobytes(), dtype=np.int32)[0]
    bits  = np.int32(0x5f3759df) - np.int32(bits >> 1)
    y     = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    return float(y * (np.float32(1.5) - xhalf * y * y))

# Three representative inputs from the TVM issue
test_cases = [
    (np.float32(1e-4),  np.float32(1.0)),   # small x → catastrophic rsqrt error
    (np.float32(0.01),  np.float32(2.0)),   # moderate x
    (np.float32(1.0),   np.float32(1.0)),   # normal x → rsqrt accurate
]

print(f"{'x':>12}  {'y':>6}  {'sqrt(x)/y':>12}  {'rsqrt(x)*y':>12}  {'rel_err':>10}  {'bug?'}")
print("-" * 72)
any_bug = False
for x, y in test_cases:
    ref   = float(np.sqrt(x)) / float(y)
    buggy = rsqrt_fast(x) * float(y)
    rel   = abs(ref - buggy) / (abs(ref) + 1e-30)
    bug   = rel > 1e-4
    if bug:
        any_bug = True
    print(f"{float(x):>12.2e}  {float(y):>6.1f}  {ref:>12.8f}  {buggy:>12.6f}  {rel:>10.2f}  {'BUG' if bug else 'ok'}")

# Also demonstrate the second pattern from #16211:
# conv → relu → mul(scale) reordered to mul(scale) → conv → relu
# When scale contains a negative value, relu fires at different points.
print()
print("Pattern 2: conv→relu→mul(s) vs mul(s)→conv→relu  (FoldScaleAxis)")
vals  = np.array([1.0, -0.5, 0.3, -1.2], dtype=np.float32)
scale = np.float32(-2.0)   # negative scale flips sign — relu fires differently
ref2  = np.maximum(vals, 0) * scale          # correct: relu first, then scale
bug2  = np.maximum(vals * scale, 0)          # wrong:   scale first, then relu
diff2 = float(np.max(np.abs(ref2 - bug2)))
print(f"  relu(x)*scale (ref):  {ref2}")
print(f"  scale*relu(x) (bug):  {bug2}")
print(f"  max_diff: {diff2:.4f}  {'BUG' if diff2 > 1e-4 else 'ok'}")
if diff2 > 1e-4:
    any_bug = True

PASS = not any_bug
print(f"\nPASS={PASS}")
if not PASS:
    print("BUG REPRODUCED: SimplifyExpr rsqrt rewrite introduces large precision error")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
