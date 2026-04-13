#!/usr/bin/env python3
"""
Bug ID     : torch_compile_cast_roundtrip
Source     : Campaign v1 (fuzzing)
Compiler   : torch.compile (Inductor)
Patterns   : cast_fp32_int32_roundtrip
Root cause : Inductor fuses the int32 cast pair into a no-op in the generated kernel, losing the floor-truncation effect on fractional values
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

# Fractional values designed to expose truncation differences near 0.5 boundaries
x = torch.tensor([0.1, 0.7, 1.4, -0.3, 2.9, -1.5, 0.5, -0.5,
                   1.9, -1.9, 3.1, -3.1, 0.99, -0.99, 1.01, -1.01],
                  dtype=torch.float32)

def cast_fn(t):
    return t.to(torch.int32).to(torch.float32)

# Eager reference
ref = cast_fn(x)

# torch.compile
try:
    compiled_cast = torch.compile(cast_fn, fullgraph=True)
    out = compiled_cast(x)
except Exception as e:
    print(f"torch.compile failed: {e}")
    print("not reproduced")
    _sys.exit(1)

diff = float(torch.max(torch.abs(out - ref)).item())
TOL = 0.01
PASS = diff <= TOL

print(f"max_diff={diff:.6f}  tol={TOL}  pass={PASS}")
print(f"ref={ref.tolist()}")
print(f"out={out.tolist()}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
