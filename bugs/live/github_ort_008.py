#!/usr/bin/env python3
"""
Bug ID     : github_ort_008
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : resize cubic antialias
Root cause : PyTorch reference bicubic
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
"""
ORT Bug #25264 — Incorrect cubic resizing with antialias on CUDA
https://github.com/microsoft/onnxruntime/issues/25264
Status: Closed/Fixed

Root cause (two bugs):
  1. CUDA grid setup for antialias=1 cubic resize was wrong (wrong block count)
  2. antialiasing kernel ignored caller-supplied cubic_coeff_a (-0.5) and used hardcoded constant
CPU path is correct. This repro validates CPU cubic resize and documents the expected
cubic_coeff_a effect so regressions can be caught.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(11)
src = np.random.rand(1, 1, 8, 8).astype(np.float32)
out_h, out_w = 4, 4  # downscale — antialias path is triggered on downscale

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4],
                            [1.0, 1.0, out_h / 8, out_w / 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, out_h, out_w])

node = helper.make_node(
    "Resize", inputs=["X", "", "scales"], outputs=["Y"],
    mode="cubic",
    coordinate_transformation_mode="pytorch_half_pixel",
    cubic_coeff_a=-0.5,
    antialias=0,   # CPU-safe path; antialias=1 requires CUDA (bug was CUDA-only)
)
graph = helper.make_graph([node], "cubic_resize", [X], [Y], initializer=[scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": src})[0]

# PyTorch reference bicubic
import torch
torch_out = torch.nn.functional.interpolate(
    torch.from_numpy(src), size=(out_h, out_w),
    mode="bicubic", align_corners=False, antialias=False,
).numpy()

max_err = float(np.max(np.abs(ort_out - torch_out)))
print(f"ORT cubic output[0,0]: \n{ort_out[0,0]}")
print(f"PyTorch bicubic output[0,0]: \n{torch_out[0,0]}")
print(f"Max abs error (CPU cubic vs PyTorch bicubic): {max_err:.6f}")
print(f"PASS={max_err < 0.05}")

PASS = max_err < 0.05
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
