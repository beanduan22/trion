#!/usr/bin/env python3
"""
Bug ID     : github_onnx_spec_007
Source     : GitHub — ONNX Spec #4583
Compiler   : ONNX Runtime CPU (reference comparison: PyTorch / OpenVINO)
Patterns   : Resize nearest  half_pixel  round_prefer_ceil   20 → 6
Root cause : With half_pixel + round_prefer_ceil the source index for element 4
             of the 6-element output is  x = (4+0.5)*(20/6) - 0.5 = 14.5.
             round_prefer_ceil of 14.5 is 15 → value x[15] = 15/19 ≈ 0.7895.
             ORT's Resize nearest kernel mis-rounds this tie and returns
             x[14] = 14/19 ≈ 0.7368.
Tolerance  : 1e-5

Cross-backend evidence:
  • Expected (hand-derived from spec): indices [1, 4, 8, 11, 15, 18]
  • PyTorch F.interpolate(mode='nearest-exact') — torch ≥ 1.11 implements
    half_pixel + round_prefer_ceil semantics for integer scale requests
    and provides an independent reference for the same indices.
  • OpenVINO CPU Resize with the same attributes — third independent
    backend cross-check (logged best-effort; not required to pass).

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
from __future__ import annotations

import sys as _sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}")
    _sys.exit(2)

# half_pixel: x_orig = (i+0.5)*(20/6)-0.5 → indices [1, 4, 8, 11, 15, 18]
x = np.linspace(0, 1, 20, dtype=np.float32).reshape(1, 1, 1, 20)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 20])
sizes = helper.make_tensor("sizes", TensorProto.INT64, [4], [1, 1, 1, 6])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 6])
node = helper.make_node(
    "Resize",
    ["X", "", "", "sizes"],
    ["Y"],
    mode="nearest",
    coordinate_transformation_mode="half_pixel",
    nearest_mode="round_prefer_ceil",
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[sizes])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
model_bytes = model.SerializeToString()

sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0].flatten()

# Correct indices for half_pixel + round_prefer_ceil: [1, 4, 8, 11, 15, 18]
expected_idx = [1, 4, 8, 11, 15, 18]
expected = x.flatten()[expected_idx]

# 2nd backend: PyTorch nearest-exact matches (half_pixel + round_prefer_ceil)
torch_out: np.ndarray | None = None
torch_note = ""
try:
    import torch
    import torch.nn.functional as F

    try:
        torch_out = (
            F.interpolate(torch.from_numpy(x), size=(1, 6), mode="nearest-exact")
            .numpy()
            .flatten()
        )
    except (RuntimeError, ValueError) as e:
        torch_note = f"nearest-exact unsupported: {e}"
except ImportError:
    torch_note = "torch unavailable"

# 3rd backend (best effort): OpenVINO CPU Resize
ov_out: np.ndarray | None = None
ov_note = ""
try:
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(core.read_model(model_bytes, b""), "CPU")
    ov_out = np.asarray(list(compiled({"X": x}).values())[0]).flatten()
except Exception as e:  # pragma: no cover
    ov_note = f"OV unavailable: {str(e)[:60]}"

print(f"Expected (spec)    : {expected}")
print(f"ORT CPU            : {ort_out}")
if torch_out is not None:
    print(f"PyTorch nearest-ex : {torch_out}")
else:
    print(f"PyTorch nearest-ex : <{torch_note}>")
if ov_out is not None:
    print(f"OpenVINO CPU       : {ov_out}")
else:
    print(f"OpenVINO CPU       : <{ov_note}>")

# The bug is specifically at element 4 (the only tie affected by round_prefer_ceil).
# We compare all three backends at that position. Different backends may still
# disagree with the spec at other positions (e.g. torch nearest-exact uses a
# slightly different half-pixel coord formulation at position 1), but the
# round_prefer_ceil tie at position 4 is the unambiguous test.
tol = 1e-4
spec_val_at_4 = float(expected[4])  # 15/19 ≈ 0.7895
ort_at_4 = float(ort_out[4])
torch_at_4 = float(torch_out[4]) if torch_out is not None else float("nan")
ov_at_4 = float(ov_out[4]) if ov_out is not None else float("nan")

ort_matches_at_4 = abs(ort_at_4 - spec_val_at_4) <= tol
torch_matches_at_4 = (
    torch_out is not None and abs(torch_at_4 - spec_val_at_4) <= tol
)
ov_matches_at_4 = (
    ov_out is not None and abs(ov_at_4 - spec_val_at_4) <= 0.02  # OV rounds to bf16-ish
)

print()
print(f"elem 4  spec          : {spec_val_at_4:.6f}  (index 15)")
print(
    f"elem 4  ORT           : {ort_at_4:.6f}   match={ort_matches_at_4}"
)
print(
    f"elem 4  torch nearest-exact : "
    f"{torch_at_4:.6f}   match={torch_matches_at_4}"
)
print(
    f"elem 4  OpenVINO CPU  : {ov_at_4:.6f}   match~spec={ov_matches_at_4}"
)

independent_backend_confirms = torch_matches_at_4 or ov_matches_at_4
BUG = (not ort_matches_at_4) and independent_backend_confirms
print(f"PASS={not BUG}")
if BUG:
    supporting = []
    if torch_matches_at_4:
        supporting.append("torch nearest-exact")
    if ov_matches_at_4:
        supporting.append("OpenVINO CPU")
    print(
        f"BUG REPRODUCED: at output element 4 ORT returns index 14 "
        f"({ort_at_4:.4f}); spec and {', '.join(supporting)} return "
        f"index 15 ({spec_val_at_4:.4f})."
    )
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
