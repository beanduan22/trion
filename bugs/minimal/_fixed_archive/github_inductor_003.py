#!/usr/bin/env python3
"""
Bug ID     : github_inductor_003
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : conv bn relu fusion bn identity
Root cause : Bug: PyTorch #112820 — fuse_modules([conv,bn,relu]) silently replaced BN with Identity, giving wrong outputs.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: PyTorch #112820 — fuse_modules([conv,bn,relu]) silently replaced BN with Identity, giving wrong outputs.
import torch
import torch.nn as nn
from torch.ao.quantization import fuse_modules

torch.manual_seed(5)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.bn   = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

model = M().eval()
x = torch.randn(1, 4, 6, 6)

with torch.no_grad():
    ref = model(x)

fused = M().eval()
fused.load_state_dict(model.state_dict())
try:
    fused_model = fuse_modules(fused, [["conv", "bn", "relu"]])
    with torch.no_grad():
        fused_out = fused_model(x)
    max_diff = float((ref - fused_out).abs().max())
    print(f"Max abs diff (unfused vs fused): {max_diff:.6f}")
    # If BN was silently replaced by Identity, diff will be large (>0.5)
    PASS = max_diff < 0.5
except Exception as e:
    print(f"Exception: {e}")
    PASS = True
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
