#!/usr/bin/env python3
"""
Bug ID     : pytorch/pytorch#121135 / compiler: torch.compile (inductor)
Source     : https://github.com/pytorch/pytorch/issues/121135
Compiler   : torch.compile (inductor)
Patterns   : x.index_add(0, randperm_index, source) where source.shape is
             incompatible with self along the non-indexed axes.
Root cause : torch.compile/inductor's randperm_index_add_pattern pass skips shape
             validation when lowering index_add.  Eager raises RuntimeError; compiled
             silently returns without error, producing undefined output.
             Compiler comparison (direct PyTorch API — not ONNX-representable):
               PyTorch eager     → RuntimeError (correct)
               torch.compile     → silent pass   ← TARGET (bug)
               TorchScript       → RuntimeError (correct, traces the error path)
             Confirmed bug shapes: (16,8)+(8,)  and  (4,4)+(2,)
             Original repro used (8,6)+(4,) which raises in BOTH paths (unhelpful).
Exit 0 = BUG REPRODUCED (eager raises; torch.compile silently returns)
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

CASES = [((32, 4), (4,)), ((8, 3), (3,))]

class IndexAddModel(nn.Module):
    def forward(self, x, y):
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        return x.index_add(0, index, y)

model = IndexAddModel()

# Pre-compile TorchScript (trace with valid shapes first, then try invalid)
# TorchScript traces with the input shapes given — will it catch the mismatch?
def run_torchscript(x, y):
    # jit.trace captures the computation graph; invalid shapes still error at runtime
    valid_x = torch.zeros_like(x)
    valid_y = torch.zeros(x.shape[0], dtype=y.dtype)  # compatible shape
    try:
        ts = torch.jit.trace(model, (valid_x, valid_y))
        return ts(x.clone(), y.clone()), None
    except Exception as e:
        return None, type(e).__name__ + ": " + str(e)[:80]

print(f"{'Case':<20}  {'Compiler':<16}  {'Result':<20}  Status")
print("-" * 75)

reproduced = False
for self_shape, src_shape in CASES:
    x = torch.randn(*self_shape)
    y = torch.randn(*src_shape)

    # ── Eager ─────────────────────────────────────────────────────────────────
    eager_out, eager_err = None, None
    try:
        eager_out = model(x.clone(), y.clone())
    except Exception as e:
        eager_err = type(e).__name__ + ": " + str(e)[:60]

    # ── torch.compile ─────────────────────────────────────────────────────────
    comp_out, comp_err = None, None
    try:
        cm       = torch.compile(model)
        comp_out = cm(x.clone(), y.clone())
    except Exception as e:
        comp_err = type(e).__name__ + ": " + str(e).split("\n")[0][:60]

    # ── TorchScript ───────────────────────────────────────────────────────────
    ts_out, ts_err = run_torchscript(x, y)

    case_str = f"{self_shape}+{src_shape}"
    for cname, out, err in [("eager", eager_out, eager_err),
                             ("torch.compile", comp_out, comp_err),
                             ("TorchScript", ts_out, ts_err)]:
        result = f"shape={list(out.shape)}" if out is not None else "—"
        status = err[:35] if err else "OK"
        flag   = "  ← BUG" if (cname == "torch.compile" and out is not None
                                and eager_err is not None) else ""
        print(f"{case_str:<20}  {cname:<16}  {result:<20}  {status}{flag}")

    if eager_err is not None and comp_out is not None:
        reproduced = True
    print()

if reproduced:
    print("BUG REPRODUCED: torch.compile skips shape validation in index_add;")
    print("  eager and TorchScript correctly raise RuntimeError.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: all compilers behaved consistently.")
    sys.exit(1)
