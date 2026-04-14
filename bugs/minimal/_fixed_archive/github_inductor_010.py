#!/usr/bin/env python3
"""
Bug ID     : github_inductor_010
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : gridsample bilinear precision
Root cause : Bug: PyTorch #81868 — GridSample bilinear identity grid: FP rounding gave [1-eps,eps,0,0] not [1,0,0,0].
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: PyTorch #81868 — GridSample bilinear identity grid: FP rounding gave [1-eps,eps,0,0] not [1,0,0,0].
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(67)
N, C, H, W = 1, 1, 4, 4
feat = np.random.randn(N, C, H, W).astype(np.float32)

# Identity grid: align_corners=True, coord = 2*i/(H-1) - 1
gy = np.linspace(-1, 1, H, dtype=np.float32)
gx = np.linspace(-1, 1, W, dtype=np.float32)
gx2d, gy2d = np.meshgrid(gx, gy)
grid = np.stack([gx2d, gy2d], axis=-1)[np.newaxis]  # [1,H,W,2]

X = helper.make_tensor_value_info("X",    TensorProto.FLOAT, [N, C, H, W])
G = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [N, H, W, 2])
Y = helper.make_tensor_value_info("Y",    TensorProto.FLOAT, None)

node  = helper.make_node("GridSample", ["X", "grid"], ["Y"],
                         mode="bilinear", padding_mode="zeros", align_corners=1)
graph = helper.make_graph([node], "g", [X, G], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out   = sess.run(None, {"X": feat, "grid": grid})[0]

max_diff = float(np.max(np.abs(out - feat)))
print(f"Identity grid bilinear max diff from input: {max_diff:.2e}")
PASS = max_diff < 1e-5
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
