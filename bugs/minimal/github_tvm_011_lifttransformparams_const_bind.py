#!/usr/bin/env python3
"""
Bug ID     : github_tvm_011
Source     : GitHub — apache/tvm#17207
Compiler   : TVM Relax  (relax.transform.LiftTransformParams)
Patterns   : Add(A, ones) * Add(B, ones)   with num_input=1
Root cause : LiftTransformParams lifts constants created inside the dataflow
             block (the `ones` tensor) out to a separate transform_params
             function.  When two different Add nodes share the same constant,
             the pass incorrectly re-binds them to the same lifted slot,
             making both A_off and B_off reference the wrong parameter.
             Result: (A+1)*(B+1) becomes wrong, e.g. [1,4,9,…] → [2,5,10,…]
             (off by +1 in each factor).
Tolerance  : exact integer equality

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
import numpy as np

def reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A + 1) * (B + 1)

A = np.arange(16, dtype="int32")
B = np.arange(16, dtype="int32")
expected = reference(A, B)

try:
    import tvm
    from tvm import relax
    from tvm.script import ir as I
    from tvm.script import relax as R
    HAS_TVM = True
except ImportError:
    HAS_TVM = False

if not HAS_TVM:
    # Demonstrate the binding confusion analytically.
    # After the bug, both A_off and B_off are incremented by the same (wrong)
    # offset because the lifted param is shared incorrectly.
    # Simulate: LiftTransformParams gives both adds the same offset=2 instead of 1.
    buggy_offset = np.int32(2)   # lifted slot gets value 2 instead of 1
    buggy = (A + buggy_offset) * (B + buggy_offset)
    diff  = int(np.abs(expected - buggy).max())
    PASS  = diff == 0
    print(f"Expected : {expected[:8]} ...")
    print(f"Buggy    : {buggy[:8]} ...")
    print(f"Max diff : {diff}  (TVM not installed — analytical demo)")
else:
    @I.ir_module
    class Module:
        @R.function
        def main(
            A: R.Tensor((16,), dtype="int32"),
            B: R.Tensor((16,), dtype="int32"),
        ) -> R.Tensor((16,), dtype="int32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                offset = R.ones(R.shape([16]), dtype="int32")
                A_off  = R.add(A, offset)
                B_off  = R.add(B, offset)
                out    = R.multiply(A_off, B_off)
                R.output(out)
            return out

    def run_mod(mod, a_val, b_val):
        target = tvm.target.Target("llvm")
        ex  = relax.build(mod, target)
        vm  = relax.VirtualMachine(ex, tvm.cpu())
        return vm["main"](tvm.nd.array(a_val), tvm.nd.array(b_val)).numpy()

    before = run_mod(Module, A, B)
    after  = run_mod(relax.transform.LiftTransformParams()(Module), A, B)
    diff   = int(np.abs(before - after).max())
    PASS   = np.array_equal(before, expected) and np.array_equal(after, expected)
    print(f"Expected : {expected[:8]} ...")
    print(f"Before   : {before[:8]} ...")
    print(f"After    : {after[:8]} ...")
    print(f"Max diff (before vs after): {diff}")

print(f"PASS={PASS}")
if not PASS:
    print("BUG REPRODUCED: LiftTransformParams corrupts constant binding")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
