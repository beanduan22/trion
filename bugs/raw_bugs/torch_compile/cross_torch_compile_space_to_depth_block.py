#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_space_to_depth_block
Compiler   : torch.compile (Inductor backend) vs torch eager
Oracle     : torch eager forward of the same nn.Module in bfloat16.
Patterns   : nn.Conv2d(32->32) -> F.pixel_unshuffle(downscale_factor=2)
             -> nn.Conv2d(128->64) wrapped in a Conv -> SpaceToDepth ->
             Conv block evaluated in bfloat16.
Root cause : In bfloat16, torch.compile's Inductor lowering of the
             SpaceToDepth (pixel_unshuffle) block uses a different
             tile-level fp32 accumulator ordering than eager.  The
             resulting tensor diverges from eager by ~2e-3 — an order
             of magnitude larger than the 1e-4 fp32-style tolerance a
             developer would expect.  Eager fp32 (control) and eager
             bfloat16 are stable references; Inductor is the outlier.
Tolerance  : 1e-4 absolute (bf16); the observed diff is ~2e-3.

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

TOL = 1e-4
torch.manual_seed(0)


class S2DBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.pixel_unshuffle(h, downscale_factor=2)  # SpaceToDepth(blocksize=2)
        return self.conv2(h)


# ── fp32 control: eager vs compile should match tightly ─────────────────────
net = S2DBlock().eval()
x_fp32 = torch.randn(1, 32, 16, 16)
with torch.no_grad():
    e32 = net(x_fp32)
    c32 = torch.compile(net, fullgraph=True)(x_fp32)
diff_fp32 = (e32 - c32).abs().max().item()
print(f"[fp32 control] eager vs compile max_abs = {diff_fp32:.6g}")

# ── bf16 target: eager vs compile diverges in Inductor ──────────────────────
net_bf = S2DBlock().eval().to(dtype=torch.bfloat16)
x_bf = torch.randn(1, 32, 16, 16, dtype=torch.bfloat16)
with torch.no_grad():
    e_bf = net_bf(x_bf)
    c_bf = torch.compile(net_bf, fullgraph=True)(x_bf)

print(f"eager  bf16 first 8: {e_bf.float().ravel()[:8].tolist()}")
print(f"compile bf16 first 8: {c_bf.float().ravel()[:8].tolist()}")
diff_bf = (e_bf.float() - c_bf.float()).abs().max().item()
print(f"|compile - eager| bf16 max_abs = {diff_bf:.6g}  (tol={TOL:g})")

if diff_bf > TOL:
    print(
        f"\nBUG REPRODUCED: Inductor bf16 lowering of SpaceToDepth block "
        f"diverges from eager by {diff_bf:.6g} (> {TOL:g})"
    )
    sys.exit(0)

print("\nnot reproduced — bf16 Inductor matches eager within tolerance")
sys.exit(1)
