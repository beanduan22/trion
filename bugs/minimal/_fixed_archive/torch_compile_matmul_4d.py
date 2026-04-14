#!/usr/bin/env python3
"""
Bug ID     : torch_compile_matmul_4d
Source     : Campaign v1 (fuzzing)
Compiler   : torch.compile (Inductor)
Patterns   : matmul_4d_batch
Root cause : Inductor generates incorrect tiling for 4D batch matmul (common in attention blocks), producing wrong values when batch dimensions exceed threshold
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""

import sys as _sys

try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

torch.manual_seed(42)
np.random.seed(42)

# Shapes: [2, 4, 8, 16] @ [2, 4, 16, 8]
x = torch.randn(2, 4, 8, 16)
w = torch.randn(2, 4, 16, 8)

def matmul_fn(a, b):
    return torch.matmul(a, b)

# Eager reference
with torch.no_grad():
    ref = matmul_fn(x, w)

# torch.compile
try:
    compiled_fn = torch.compile(matmul_fn, fullgraph=True)
    with torch.no_grad():
        out = compiled_fn(x, w)
except Exception as e:
    print(f"torch.compile failed: {e}")
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
