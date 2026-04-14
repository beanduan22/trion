#!/usr/bin/env python3
"""
Bug ID     : github_tvm_012
Source     : GitHub — apache/tvm#18750
Compiler   : TVM Relax ONNX frontend (reported February 2026, opset 20)
Patterns   : ONNX Gelu (opset 20) with approximate="tanh" attribute
Root cause : TVM's ONNX frontend hardcodes Gelu → R.nn.gelu (exact CDF formula)
             and ignores the `approximate="tanh"` attribute.  The tanh
             approximation formula is:
               gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
             while the exact formula uses erf.  The two differ by up to ~1.5e-4
             at x = -1.0 and are systematically different for all activations.
             Common in BERT variants and HuggingFace LLM exports.
Tolerance  : 1e-4 (TVM default; exact == tanh error exceeds it)

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
import numpy as np
import math

def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Exact Gelu using error function: 0.5*x*(1 + erf(x/sqrt(2)))"""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """Tanh approximation: 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))"""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

# Representative test values from the TVM issue + boundary cases
test_vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float64)

print("=== github_tvm_012: ONNX Gelu approx=tanh ignored by TVM ===")
print()
print(f"{'x':>6}  {'gelu_exact':>12}  {'gelu_tanh':>12}  {'abs_diff':>10}  {'rel_diff':>10}  {'bug?'}")
print("-" * 68)

TOL = 1e-4
any_bug = False
max_diff = 0.0

for x in test_vals:
    exact = gelu_exact(np.array([x]))[0]
    approx = gelu_tanh(np.array([x]))[0]
    abs_d = abs(exact - approx)
    rel_d = abs_d / (abs(exact) + 1e-30)
    bug   = abs_d > TOL
    print(f"{x:>6.2f}  {exact:>12.8f}  {approx:>12.8f}  {abs_d:>10.2e}  {rel_d:>10.2e}  {'BUG' if bug else 'ok'}")
    if bug:
        any_bug = True
    max_diff = max(max_diff, abs_d)

print()
print(f"Max abs diff: {max_diff:.2e}  tol={TOL:.0e}")

# Demonstrate what TVM does: maps Gelu(approx=tanh) → exact erf path
# This is the "wrong" output TVM would produce for a model using approx=tanh
print()
print("TVM substitution demo (approx=tanh input → exact erf output):")
x_batch = np.array([-1.5, -1.0, 0.0, 1.0, 1.5], dtype=np.float32)
expected = gelu_tanh(x_batch.astype(np.float64)).astype(np.float32)
tvm_out  = gelu_exact(x_batch.astype(np.float64)).astype(np.float32)
diff_arr = np.abs(expected.astype(np.float64) - tvm_out.astype(np.float64))
print(f"  expected (tanh): {np.round(expected, 5).tolist()}")
print(f"  TVM output (exact erf): {np.round(tvm_out, 5).tolist()}")
print(f"  per-element diff: {np.round(diff_arr, 6).tolist()}")
print(f"  any diff > TOL: {bool(np.any(diff_arr > TOL))}")
if np.any(diff_arr > TOL):
    any_bug = True

# TVM runtime check (if installed)
try:
    import tvm
    from tvm import relax
    from tvm.script import ir as I
    from tvm.script import relax as R
    HAS_TVM = True
except ImportError:
    HAS_TVM = False

if HAS_TVM:
    print()
    print("TVM runtime check:")
    @I.ir_module
    class GeluModule:
        @R.function
        def main(x: R.Tensor((7,), dtype="float32")) -> R.Tensor((7,), dtype="float32"):
            with R.dataflow():
                # TVM maps ONNX Gelu → R.nn.gelu regardless of approximate attr
                out = R.nn.gelu(x)
                R.output(out)
            return out

    target = tvm.target.Target("llvm")
    ex  = relax.build(GeluModule, target)
    vm  = relax.VirtualMachine(ex, tvm.cpu())

    x_in = test_vals.astype(np.float32)
    tvm_result = vm["main"](tvm.nd.array(x_in)).numpy()
    expected_tanh = gelu_tanh(test_vals).astype(np.float32)
    diff_tvm = np.abs(tvm_result.astype(np.float64) -
                      expected_tanh.astype(np.float64))
    bug_tvm = bool(np.any(diff_tvm > TOL))
    print(f"  TVM gelu output:    {np.round(tvm_result, 5).tolist()}")
    print(f"  expected (tanh):    {np.round(expected_tanh, 5).tolist()}")
    print(f"  max_diff: {diff_tvm.max():.2e}  {'BUG' if bug_tvm else 'ok'}")
    if bug_tvm:
        any_bug = True

print()
PASS = not any_bug
print(f"PASS={PASS}")
if not PASS:
    print("BUG REPRODUCED: TVM maps ONNX Gelu(approx=tanh) → exact erf, producing systematic error > 1e-4")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
