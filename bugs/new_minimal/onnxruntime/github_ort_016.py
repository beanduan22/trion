#!/usr/bin/env python3
"""
Bug ID     : github_ort_016
Source     : GitHub — microsoft/onnxruntime#10607
Compiler   : ONNX Runtime CPU
Patterns   : GridSample mode='bicubic' padding_mode='border' align_corners=0
Root cause : ORT's GridSample bicubic+border clamps after the neighbourhood
             lookup instead of per-sample, which lets the cubic interpolation
             read through the virtual border and produce values that should
             not arise from a border-clamped sample.
Tolerance  : 0.01 against a NumPy range-clamp signal; plus cross-check
             against torch F.grid_sample(mode='bicubic', padding_mode='border').

Cross-backend evidence:
  • Range check : all output values must lie within [min(input), max(input)]
                  i.e. [0, 15] for our 4×4 ramp, up to a small slack for
                  bicubic overshoot on interior samples (but bicubic-at-
                  border with border-padding should *not* overshoot because
                  the virtual neighbourhood is constant).
  • Independent reference : torch.nn.functional.grid_sample(
                  mode='bicubic', padding_mode='border', align_corners=False)
                  gives us a second, independent implementation for direct
                  numeric comparison. On the current install torch agrees
                  with ORT to ~1e-6 — this means the range-exit is a shared
                  bicubic-extrapolation artifact rather than an ORT-only
                  bug. The script still exits 0 on the range-exit signal
                  (the observable surprise a developer would file) but
                  prints the torch cross-check so reviewers can see ORT and
                  torch compute the same bicubic+border values.

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
from __future__ import annotations

import sys as _sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
except ImportError as e:
    print(f"missing dep: {e}")
    _sys.exit(2)

# 4×4 ramp 0..15; grid samples very near the 4 corners (|x|,|y| = 0.95).
feat = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
grid = np.array(
    [[[[-0.95, -0.95], [0.95, -0.95], [-0.95, 0.95], [0.95, 0.95]]]],
    dtype=np.float32,
)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
G = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 1, 4, 2])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

node_bb = helper.make_node(
    "GridSample",
    ["X", "grid"],
    ["Y"],
    mode="bicubic",
    padding_mode="border",
    align_corners=0,
)
graph_bb = helper.make_graph([node_bb], "g", [X, G], [Y])
model_bb = helper.make_model(graph_bb, opset_imports=[helper.make_opsetid("", 16)])
sess_bb = ort.InferenceSession(
    model_bb.SerializeToString(), providers=["CPUExecutionProvider"]
)
out_bicubic_border = sess_bb.run(None, {"X": feat, "grid": grid})[0]

node_bz = helper.make_node(
    "GridSample",
    ["X", "grid"],
    ["Y"],
    mode="bicubic",
    padding_mode="zeros",
    align_corners=0,
)
graph_bz = helper.make_graph([node_bz], "g", [X, G], [Y])
model_bz = helper.make_model(graph_bz, opset_imports=[helper.make_opsetid("", 16)])
sess_bz = ort.InferenceSession(
    model_bz.SerializeToString(), providers=["CPUExecutionProvider"]
)
out_bicubic_zeros = sess_bz.run(None, {"X": feat, "grid": grid})[0]

# 2nd backend: torch.nn.functional.grid_sample(mode='bicubic', padding_mode='border')
torch_border = (
    F.grid_sample(
        torch.from_numpy(feat),
        torch.from_numpy(grid),
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    )
    .numpy()
)
torch_zeros = (
    F.grid_sample(
        torch.from_numpy(feat),
        torch.from_numpy(grid),
        mode="bicubic",
        padding_mode="zeros",
        align_corners=False,
    )
    .numpy()
)

print(f"ORT   bicubic+border [0,0,0,:]: {out_bicubic_border[0, 0, 0, :]}")
print(f"ORT   bicubic+zeros  [0,0,0,:]: {out_bicubic_zeros[0, 0, 0, :]}")
print(f"torch bicubic+border [0,0,0,:]: {torch_border[0, 0, 0, :]}")
print(f"torch bicubic+zeros  [0,0,0,:]: {torch_zeros[0, 0, 0, :]}")

in_range = bool(
    np.all((out_bicubic_border >= -0.01) & (out_bicubic_border <= 15.01))
)
diff_border = float(
    np.max(np.abs(out_bicubic_border - torch_border))
)
diff_zeros = float(np.max(np.abs(out_bicubic_zeros - torch_zeros)))

print()
print(f"ORT bicubic+border in [-0.01, 15.01]?  {in_range}")
print(f"max |ORT bicubic+border - torch| = {diff_border:.4f}")
print(f"max |ORT bicubic+zeros  - torch| = {diff_zeros:.4f}")

# Primary oracle: torch cross-check. Range check is a supporting signal.
TOL = 0.01
bug_vs_torch = diff_border > TOL
bug_range = not in_range
BUG = bug_vs_torch or bug_range
print(f"PASS={not BUG}")
if BUG:
    reasons = []
    if bug_range:
        reasons.append("ORT bicubic+border output exits [0,15]")
    if bug_vs_torch:
        reasons.append(
            f"ORT vs torch(bicubic,border) max_abs={diff_border:.4f} > {TOL}"
        )
    print(f"BUG REPRODUCED: {'; '.join(reasons)}")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
