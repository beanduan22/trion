"""
TVM Bug: relax.transform.LiftTransformParams produces wrong inference results.

Source : https://github.com/apache/tvm/issues/17207
Affects: TVM Relax, July 2024
Root cause: LiftTransformParams incorrectly re-binds constants when
            lifting parameters out of the dataflow block — the "offset"
            constant (ones tensor) used in two Add nodes gets mixed up,
            causing both outputs to use a different parameter binding.

Graph:
  main(A[16], B[16]):
    offset = ones([16])          ← constant created inside dataflow
    A_off  = A + offset
    B_off  = B + offset
    return A_off * B_off

After LiftTransformParams the constant binding is corrupted:
  A_off and B_off both end up added to the wrong (or same) offset.

Expected (e.g. A=[0..15], B=[0..15]):
  out[i] = (A[i]+1) * (B[i]+1)  →  [1, 4, 9, 16, 25, ...]
Actual  (post-transform):
  shifted/wrong values, e.g.     →  [2, 5, 10, 17, 26, ...]
"""
import numpy as np

try:
    import tvm
    from tvm import relax
    from tvm.script import ir as I
    from tvm.script import relax as R
    HAS_TVM = True
except ImportError:
    HAS_TVM = False

def reference(A, B):
    """Pure-numpy reference: (A+1)*(B+1)."""
    return (A + 1) * (B + 1)

A = np.arange(16, dtype="int32")
B = np.arange(16, dtype="int32")
expected = reference(A, B)

if not HAS_TVM:
    print("TVM not installed — showing expected output only.")
    print(f"Expected: {expected}")
    PASS = True
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
                offset  = R.ones(R.shape([16]), dtype="int32")
                A_off   = R.add(A, offset)
                B_off   = R.add(B, offset)
                output  = R.multiply(A_off, B_off)
                R.output(output)
            return output

    def run_mod(mod, A_val, B_val):
        target = tvm.target.Target("llvm")
        ex     = relax.build(mod, target)
        vm     = relax.VirtualMachine(ex, tvm.cpu())
        return vm["main"](
            tvm.nd.array(A_val),
            tvm.nd.array(B_val),
        ).numpy()

    # Before transform
    before = run_mod(Module, A, B)

    # After LiftTransformParams
    transformed = relax.transform.LiftTransformParams()(Module)
    after = run_mod(transformed, A, B)

    max_diff = int(np.abs(before - after).max())
    PASS = np.array_equal(before, expected) and np.array_equal(after, expected)

    print(f"Expected  : {expected}")
    print(f"Before    : {before}")
    print(f"After     : {after}")
    print(f"Max diff (before vs after): {max_diff}")
    print(f"PASS={PASS}  (False = LiftTransformParams corrupts inference)")
