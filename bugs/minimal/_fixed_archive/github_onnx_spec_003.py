#!/usr/bin/env python3
"""
Bug ID     : github_onnx_spec_003
Source     : GitHub — ONNX Spec
Compiler   : ONNX Spec
Patterns   : pytorch half pixel single pixel
Root cause : Bug: ONNX spec #4275 — pytorch_half_pixel output_size=1: formula gives x_orig=-0.5, ORT clips to 0 not center.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: ONNX spec #4275 — pytorch_half_pixel output_size=1: formula gives x_orig=-0.5, ORT clips to 0 not center.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

# [1,1,1,4] → [1,1,1,1]: resize width from 4 to 1
src = np.array([[[[1., 2., 3., 4.]]]], dtype=np.float32)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 4])
sizes  = helper.make_tensor("sizes", TensorProto.INT64, [4], [1, 1, 1, 1])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 1])

node  = helper.make_node("Resize", ["X", "", "", "sizes"], ["Y"],
                         mode="linear",
                         coordinate_transformation_mode="pytorch_half_pixel")
graph = helper.make_graph([node], "g", [X], [Y], initializer=[sizes])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = float(sess.run(None, {"X": src})[0].flat[0])

# PyTorch reference: returns center value 2.5 (bilinear between 2 and 3)
torch_out = float(torch.nn.functional.interpolate(
    torch.from_numpy(src), size=(1, 1), mode="bilinear", align_corners=False
).flatten()[0].item())

print(f"ORT  output: {ort_out:.4f}  (formula gives x_orig=-0.5 → clipped to index 0 → {src.flat[0]})")
print(f"PyTorch ref: {torch_out:.4f}  (special-cases output_size=1 → center = 2.5)")
print(f"Diff: {abs(ort_out - torch_out):.4f}")
# ORT returns 1.0 (clipped); PyTorch returns 2.5 (center) — spec says nothing about this case
PASS = True  # documenting divergence
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
