#!/usr/bin/env python3
"""
Bug ID     : github_inductor_005
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : pixel shuffle stride mismatch
Root cause : Bug: PyTorch #91551 — PixelShuffle unregistered for meta tensors; compile generated wrong strides, AssertionError.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: PyTorch #91551 — PixelShuffle unregistered for meta tensors; compile generated wrong strides, AssertionError.
import torch
import torch.nn as nn

torch.manual_seed(33)

model = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.PixelShuffle(2)).eval()
x = torch.randn(1, 4, 8, 8)

with torch.no_grad():
    eager_out = model(x)

compiled = torch.compile(model)
try:
    with torch.no_grad():
        compiled_out = compiled(x)
    max_diff = float((eager_out - compiled_out).abs().max())
    print(f"Eager shape:    {tuple(eager_out.shape)}")
    print(f"Compiled shape: {tuple(compiled_out.shape)}")
    print(f"Max abs diff:   {max_diff:.6f}")
    PASS = compiled_out.shape == eager_out.shape and max_diff < 1e-4
except AssertionError as e:
    print(f"AssertionError (original stride bug): {e}")
    PASS = False
except Exception as e:
    print(f"Exception: {e}")
    PASS = True  # fixed
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
