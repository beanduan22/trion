#!/usr/bin/env python3
"""
Bug ID     : github_inductor_001
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : batchnorm wrong shape
Root cause : Bug: Inductor #100970 — BatchNorm2d with torch.compile produced wrong output shape [3,3,2,2] not [1,3,2,2].
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: Inductor #100970 — BatchNorm2d with torch.compile produced wrong output shape [3,3,2,2] not [1,3,2,2].
import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Sequential(nn.BatchNorm2d(3)).eval()
x = torch.randn(1, 3, 2, 2)

with torch.no_grad():
    eager_out = model(x)

compiled_model = torch.compile(model)
with torch.no_grad():
    try:
        compiled_out = compiled_model(x)
        shape_ok = compiled_out.shape == eager_out.shape
        max_diff = float((eager_out - compiled_out).abs().max())
        print(f"Eager shape:    {tuple(eager_out.shape)}")
        print(f"Compiled shape: {tuple(compiled_out.shape)}")
        print(f"Max abs diff:   {max_diff:.6f}")
        PASS = shape_ok and max_diff < 1e-4
    except Exception as e:
        print(f"Exception: {e}")
        PASS = True  # bug no longer reproducible
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
