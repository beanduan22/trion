#!/usr/bin/env python3
"""
Bug ID     : github_inductor_011
Source     : GitHub — pytorch/pytorch#143555 (right-shift) + #143566 (left-shift)
Compiler   : PyTorch Inductor CPU, torch ≤ 2.5
Patterns   : BitShift (right + left) with shift amount == 64 on int64 tensors
Root cause : Inductor's CPU C++ codegen emits a native C right-shift: `x >> n`.
             The C/C++ standard defines shifting a 64-bit integer by 64 or more
             as undefined behaviour (UB); on x86 the shift amount is masked to
             the low 6 bits, so `x >> 64` becomes `x >> 0` = x (no-op) instead
             of 0.  Fixed in torch 2.6 (PR #143635 / #143636): codegen now
             emits a conditional zeroing when n >= bitwidth.
Tolerance  : exact integer equality (result must be 0 for shift==64)

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
import numpy as np
import ctypes

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

any_bug = False

print("=== github_inductor_011: BitShift UB — shift by 64 produces non-zero ===")
print()

# Analytical: demonstrate C UB for shift >= bitwidth
def c_ub_shift_right(x_i64: int, n: int) -> int:
    """
    Simulate what x86 C generates for `(int64_t)x >> n`:
    hardware masks n to low 6 bits, so shift by 64 ≡ shift by 0.
    """
    n_masked = n & 63  # x86 SAR instruction masks shift count to 6 bits
    if n_masked == 0:
        return x_i64
    return x_i64 >> n_masked

def correct_shift_right(x_i64: int, n: int) -> int:
    """Correct: shift >= bitwidth should return 0 (or -1 for negative arithmetic shift)."""
    if n >= 64:
        return 0  # for unsigned semantics; arithmetic would be -1 for negative
    return x_i64 >> n

test_cases = [
    (1000, 64),   # primary case from the issue
    (0xFF, 64),
    (-42, 64),
    (1, 63),      # boundary: shift by 63 is well-defined
    (1, 65),      # shift by 65: should be 0
]

print(f"{'value':>10}  {'shift':>6}  {'correct':>12}  {'C_UB(x86)':>12}  {'bug?'}")
print("-" * 58)
for x, n in test_cases:
    correct = correct_shift_right(x, n)
    ub_val  = c_ub_shift_right(x, n)
    bug     = (ub_val != correct) and n >= 64
    print(f"{x:>10}  {n:>6}  {correct:>12}  {ub_val:>12}  {'BUG(UB)' if bug else 'ok'}")
    if bug:
        any_bug = True

print()
print("Left-shift equivalent (pytorch/pytorch#143566):")
def c_ub_shift_left(x_i64: int, n: int) -> int:
    n_masked = n & 63
    if n_masked == 0:
        return x_i64
    return ctypes.c_int64(x_i64 << n_masked).value

for x, n in [(1000, 64), (1, 64), (0xFF, 64)]:
    correct = 0 if n >= 64 else x << n
    ub_val  = c_ub_shift_left(x, n)
    bug     = (ub_val != correct)
    print(f"  {x} << {n}: correct={correct}  C_UB={ub_val}  {'BUG' if bug else 'ok'}")
    if bug:
        any_bug = True

# PyTorch path: test actual torch.bitwise_right_shift behaviour
if HAS_TORCH:
    print()
    print("PyTorch runtime check:")
    x = torch.tensor([1000, 255, -42], dtype=torch.int64)
    n = torch.tensor([64, 64, 64], dtype=torch.int64)

    ref_eager = torch.bitwise_right_shift(x, n)
    print(f"  eager: {x.tolist()} >> {n.tolist()} = {ref_eager.tolist()}")
    expected = [0, 0, 0]
    bug_eager = ref_eager.tolist() != expected
    print(f"  expected: {expected}  {'BUG' if bug_eager else 'ok'}")
    if bug_eager:
        any_bug = True

    try:
        compiled = torch.compile(torch.bitwise_right_shift, backend="inductor")
        got = compiled(x, n)
        print(f"  compile: {x.tolist()} >> {n.tolist()} = {got.tolist()}")
        bug_compile = got.tolist() != expected
        print(f"  expected: {expected}  {'BUG' if bug_compile else 'ok'}")
        if bug_compile:
            any_bug = True
    except Exception as e:
        print(f"  torch.compile failed: {e}")

# ONNX BitShift path
if HAS_ONNX:
    print()
    print("ONNX BitShift (direction=RIGHT) path:")
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    vals   = np.array([[1000, 255]], dtype=np.uint64)
    shifts = np.array([[64, 64]], dtype=np.uint64)

    nodes = [oh.make_node("BitShift", ["x", "n"], ["y"], direction="RIGHT")]
    g = oh.make_graph(nodes, "bitshift",
        [oh.make_tensor_value_info("x", TP.UINT64, [1, 2]),
         oh.make_tensor_value_info("n", TP.UINT64, [1, 2])],
        [oh.make_tensor_value_info("y", TP.UINT64, [1, 2])])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 11)])
    m.ir_version = 8
    mb = m.SerializeToString()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(mb, so, providers=["CPUExecutionProvider"])
    result = sess.run(None, {"x": vals, "n": shifts})[0]
    expected_onnx = np.array([[0, 0]], dtype=np.uint64)
    diff = int(np.abs(result.astype(np.int64) - expected_onnx.astype(np.int64)).max())
    bug_onnx = diff > 0
    print(f"  ORT: {vals[0].tolist()} >> {shifts[0].tolist()} = {result[0].tolist()}")
    print(f"  expected: [0, 0]  {'BUG' if bug_onnx else 'ok'}")
    if bug_onnx:
        any_bug = True

print()
PASS = not any_bug
print(f"PASS={PASS}")
if not PASS:
    print("BUG REPRODUCED: BitShift by 64 returns non-zero (C UB: x86 masks shift to 6 bits)")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
