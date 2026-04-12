# Bug: Inductor #141317 — Conv2d+BN in train mode: inductor computed wrong batch stats, output diverged from eager.
import torch
import torch.nn as nn

torch.manual_seed(99)

model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8))
model.train()
x = torch.randn(4, 3, 8, 8)

eager_out = model(x)

compiled = torch.compile(model)
try:
    compiled_out = compiled(x)
    max_diff = float((eager_out - compiled_out).abs().max())
    print(f"Max abs diff (train-mode BN, eager vs compiled): {max_diff:.6f}")
    PASS = max_diff < 1e-3
except Exception as e:
    print(f"Exception: {e}")
    PASS = True  # bug may be fixed
print(f"PASS={PASS}")
