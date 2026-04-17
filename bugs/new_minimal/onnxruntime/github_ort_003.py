#!/usr/bin/env python3
"""
Bug ID     : github_ort_003
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : resize linear bilinear error
Root cause : Bug: asymmetric coord mode gives up to 0.2947 error vs PyTorch bilinear.
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

# Bug: asymmetric coord mode gives up to 0.2947 error vs PyTorch bilinear.
np.random.seed(42)
src = np.random.rand(1, 1, 4, 4).astype(np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.5, 2.5])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 10, 10])
node = helper.make_node(
    "Resize", inputs=["X", "", "scales"], outputs=["Y"],
    mode="linear", coordinate_transformation_mode="asymmetric",
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": src})[0].flatten()

torch_out = torch.nn.functional.interpolate(
    torch.from_numpy(src), size=(10, 10), mode="bilinear", align_corners=False,
).numpy().flatten()

max_err = float(np.max(np.abs(ort_out - torch_out)))
print(f"asymmetric coord mode max abs error vs PyTorch: {max_err:.4f}")
print(f"ORT   output[:4]: {ort_out[:4]}")
print(f"Torch output[:4]: {torch_out[:4]}")
PASS = (max_err < 0.01)
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
