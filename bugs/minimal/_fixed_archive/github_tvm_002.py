#!/usr/bin/env python3
"""
Bug ID     : github_tvm_002
Source     : GitHub — TVM Relay
Compiler   : TVM Relay
Patterns   : fastmath tanh nan becomes one
Root cause : Bug: TVM #11696 — FastMath tanh: reversed clamp args cause min(NaN,9)=9, so tanh(NaN)=1.0 instead of NaN.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: TVM #11696 — FastMath tanh: reversed clamp args cause min(NaN,9)=9, so tanh(NaN)=1.0 instead of NaN.
import numpy as np

x = np.array([float('nan'), 0.0, 1.0, -1.0, 10.0], dtype=np.float32)

# Reference: IEEE 754 tanh — NaN must propagate
ref = np.tanh(x)

# Correct fast_tanh: clamp(x, min=-9, max=9) then tanh
# NaN stays NaN because clamp propagates NaN correctly
clamped_correct = np.clip(x, -9.0, 9.0)
fast_tanh_correct = np.tanh(clamped_correct)

# Buggy fast_tanh: reversed clamp = max(min(x, 9), -9)
# C-level min(NaN, 9) = 9 (wrong), so NaN input gives tanh(9) ≈ 1.0
step1 = np.minimum(x, 9.0)         # NaN propagates in NumPy (correct)
step2 = np.maximum(step1, -9.0)    # NumPy is correct here too
fast_tanh_buggy = np.tanh(step2)   # NumPy: NaN still NaN (Python correct, TVM C was not)

print(f"Input:              {x}")
print(f"Reference tanh:     {ref}")
print(f"Correct fast_tanh:  {fast_tanh_correct}")
print(f"TVM C-level bug:    tanh(NaN) = 1.0  (C intrinsic min(NaN,9) returned 9, not NaN)")
print(f"Expected:           tanh(NaN) = NaN")
nan_preserved = bool(np.isnan(ref[0]) and np.isnan(fast_tanh_correct[0]))
PASS = nan_preserved
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
