#!/usr/bin/env python3
"""
Bug ID     : github_ort_016
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : gridsample bicubic border
Root cause : Bug: ORT GridSample bicubic+border clamps after neighbourhood lookup instead of per-sample (issue #10607).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: ORT GridSample bicubic+border clamps after neighbourhood lookup instead of per-sample (issue #10607).
feat = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
grid = np.array([[[[-0.95, -0.95], [0.95, -0.95],
                   [-0.95,  0.95], [0.95,  0.95]]]], dtype=np.float32)

X = helper.make_tensor_value_info("X",    TensorProto.FLOAT, [1, 1, 4, 4])
G = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 1, 4, 2])
Y = helper.make_tensor_value_info("Y",    TensorProto.FLOAT, None)

node_bb = helper.make_node("GridSample", ["X","grid"], ["Y"],
    mode="bicubic", padding_mode="border", align_corners=0)
graph_bb = helper.make_graph([node_bb], "g", [X, G], [Y])
model_bb = helper.make_model(graph_bb, opset_imports=[helper.make_opsetid("", 16)])
sess_bb  = ort.InferenceSession(model_bb.SerializeToString(), providers=["CPUExecutionProvider"])
out_bicubic_border = sess_bb.run(None, {"X": feat, "grid": grid})[0]

node_bz = helper.make_node("GridSample", ["X","grid"], ["Y"],
    mode="bicubic", padding_mode="zeros", align_corners=0)
graph_bz = helper.make_graph([node_bz], "g", [X, G], [Y])
model_bz = helper.make_model(graph_bz, opset_imports=[helper.make_opsetid("", 16)])
sess_bz  = ort.InferenceSession(model_bz.SerializeToString(), providers=["CPUExecutionProvider"])
out_bicubic_zeros = sess_bz.run(None, {"X": feat, "grid": grid})[0]

print(f"bicubic+border output[0,0,0,:]: {out_bicubic_border[0,0,0,:]}")
print(f"bicubic+zeros  output[0,0,0,:]: {out_bicubic_zeros[0,0,0,:]}")
in_range = bool(np.all((out_bicubic_border >= -0.01) & (out_bicubic_border <= 15.01)))
print(f"Bicubic+border values in [0,15] range: {in_range}")
PASS = in_range
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
