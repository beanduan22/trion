#!/usr/bin/env python3
"""
Bug ID     : torch_compile_avgpool_ceil
Source     : Campaign v1 (fuzzing)
Compiler   : torch.compile (Inductor)
Patterns   : ceil_mode_avg_pool_conv
Root cause : torch.compile Inductor uses floor-mode padding calculation for the fused AvgPool+Conv kernel when ceil_mode=True, producing wrong output size or values
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""

import sys as _sys

try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

torch.manual_seed(42)
np.random.seed(42)

# Input [1, 4, 7, 7]
INPUT_SHAPE = (1, 4, 7, 7)

avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
conv     = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, bias=False)

# Seed conv weights for reproducibility
nn.init.normal_(conv.weight, mean=0.0, std=0.1)

def model_fn(x):
    return conv(avg_pool(x))

x = torch.randn(*INPUT_SHAPE)

# Eager reference
with torch.no_grad():
    ref = model_fn(x)

# torch.compile
try:
    compiled_fn = torch.compile(model_fn, fullgraph=True)
    with torch.no_grad():
        out = compiled_fn(x)
except Exception as e:
    print(f"torch.compile failed: {e}")
    # If compile itself fails, treat as not reproduced
    print("not reproduced")
    _sys.exit(1)

diff = float(torch.max(torch.abs(out - ref)).item())
TOL = 0.01
PASS = diff <= TOL

print(f"max_diff={diff:.6f}  tol={TOL}  pass={PASS}")
print(f"ref shape={list(ref.shape)}  out shape={list(out.shape)}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
