#!/usr/bin/env python3
"""
Bug ID     : github_inductor_009
Source     : GitHub — torch.compile (Inductor)
Compiler   : torch.compile (Inductor)
Patterns   : gridsample nan inf gpu
Root cause : Bug: PyTorch #24823 — GridSample NaN grid coords corrupted neighbouring valid outputs on GPU.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# Bug: PyTorch #24823 — GridSample NaN grid coords corrupted neighbouring valid outputs on GPU.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

np.random.seed(61)
feat = np.random.randn(1, 1, 6, 6).astype(np.float32)

# Grid with mixed valid and NaN coordinates [N=1, H_out=1, W_out=4, 2]
grid = np.array([[[[0.0, 0.0],
                   [float('nan'), 0.0],
                   [0.5, 0.5],
                   [0.0, float('nan')]]]], dtype=np.float32)

X = helper.make_tensor_value_info("X",    TensorProto.FLOAT, [1, 1, 6, 6])
G = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 1, 4, 2])
Y = helper.make_tensor_value_info("Y",    TensorProto.FLOAT, None)

node  = helper.make_node("GridSample", ["X", "grid"], ["Y"],
                         mode="bilinear", padding_mode="zeros", align_corners=1)
graph = helper.make_graph([node], "g", [X, G], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
sess  = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out   = sess.run(None, {"X": feat, "grid": grid})[0]

# Valid inputs (index 0 and 2) must not be NaN
valid_ok = not np.isnan(out[0, 0, 0, 0]) and not np.isnan(out[0, 0, 0, 2])
# NaN inputs (index 1 and 3) should produce NaN
nan_ok   = np.isnan(out[0, 0, 0, 1]) and np.isnan(out[0, 0, 0, 3])
print(f"Output: {out[0, 0, 0, :]}")
print(f"Valid outputs not NaN: {valid_ok}")
print(f"NaN inputs → NaN output: {nan_ok}")
PASS = valid_ok
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
