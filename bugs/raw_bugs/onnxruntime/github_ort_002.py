#!/usr/bin/env python3
"""
Bug ID     : github_ort_002
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : resize nearest one pixel off
Root cause : Bug: nearest resize 1->64 (scale=64/26) produces pixel selection one off vs PyTorch.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

# Bug: nearest resize 1->64 (scale=64/26) produces pixel selection one off vs PyTorch.
src = np.arange(26, dtype=np.float32).reshape(1, 1, 1, 26)
scales_val = np.array([1.0, 1.0, 1.0, 64.0 / 26.0], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 26])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], scales_val.tolist())
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 64])
node = helper.make_node(
    "Resize", inputs=["X", "", "scales"], outputs=["Y"],
    mode="nearest", coordinate_transformation_mode="half_pixel",
    nearest_mode="round_prefer_ceil",
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": src})[0].flatten()

torch_out = torch.nn.functional.interpolate(
    torch.from_numpy(src), size=(1, 64), mode="nearest"
).numpy().flatten()

mismatch = int(np.sum(ort_out != torch_out))
print(f"ORT   output[30:35]: {ort_out[30:35]}")
print(f"Torch output[30:35]: {torch_out[30:35]}")
print(f"Mismatched elements: {mismatch}/64")
PASS = (mismatch == 0)
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
