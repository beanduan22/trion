#!/usr/bin/env python3
"""
Bug ID     : github_inductor_004
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : interpolate bilinear wrong
Root cause : Bug: Inductor #93262 — torch.compile AOT decomposition of bilinear interpolate silently gave wrong values.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: Inductor #93262 — torch.compile AOT decomposition of bilinear interpolate silently gave wrong values.
import torch
import torch.nn.functional as F

torch.manual_seed(21)

x = torch.randn(1, 3, 8, 8)

def fn(t):
    return F.interpolate(t, size=(16, 16), mode="bilinear", align_corners=False)

eager_out = fn(x)

compiled_fn = torch.compile(fn)
try:
    compiled_out = compiled_fn(x)
    max_diff = float((eager_out - compiled_out).abs().max())
    print(f"Max abs diff (bilinear, eager vs compiled): {max_diff:.6f}")
    PASS = max_diff < 1e-3
except Exception as e:
    print(f"Exception: {e}")
    PASS = True  # bug fixed
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
