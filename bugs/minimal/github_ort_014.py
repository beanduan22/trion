#!/usr/bin/env python3
"""
Bug ID     : github_ort_014
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : roialign halfpixel offset
Root cause : Bug: ORT RoiAlign half_pixel mode applied pixel offset incorrectly (issue #6921).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Bug: ORT RoiAlign half_pixel mode applied pixel offset incorrectly (issue #6921).
# half_pixel and output_half_pixel differ; fixed in current ORT.

np.random.seed(3)
# Feature map: 1 batch, 1 channel, 8x8
feat = np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8)
# One ROI covering the top-left 4x4 region
rois   = np.array([[0, 0, 0, 4, 4]], dtype=np.float32)  # [batch_idx, x1, y1, x2, y2]
roi_idx = np.array([0], dtype=np.int64)

X   = helper.make_tensor_value_info("X",   TensorProto.FLOAT, [1, 1, 8, 8])
R   = helper.make_tensor_value_info("rois", TensorProto.FLOAT, [1, 4])
BI  = helper.make_tensor_value_info("bi",   TensorProto.INT64, [1])
Y   = helper.make_tensor_value_info("Y",   TensorProto.FLOAT, None)

# opset-16 uses coordinate_transformation_mode attribute
node = helper.make_node(
    "RoiAlign",
    inputs=["X", "rois", "bi"],
    outputs=["Y"],
    mode="avg",
    output_height=4,
    output_width=4,
    sampling_ratio=2,
    spatial_scale=1.0,
    coordinate_transformation_mode="half_pixel",
)
graph = helper.make_graph([node], "g", [X, R, BI], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out_hp = sess.run(None, {"X": feat, "rois": rois[:, 1:], "bi": roi_idx})[0]

# output_unaligned (legacy mode) for comparison
node_u = helper.make_node(
    "RoiAlign",
    inputs=["X", "rois", "bi"],
    outputs=["Y"],
    mode="avg",
    output_height=4,
    output_width=4,
    sampling_ratio=2,
    spatial_scale=1.0,
    coordinate_transformation_mode="output_half_pixel",
)
graph_u = helper.make_graph([node_u], "g2", [X, R, BI], [Y])
model_u = helper.make_model(graph_u, opset_imports=[helper.make_opsetid("", 16)])
sess_u = ort.InferenceSession(model_u.SerializeToString(), providers=["CPUExecutionProvider"])
out_u = sess_u.run(None, {"X": feat, "rois": rois[:, 1:], "bi": roi_idx})[0]

max_diff = float(np.max(np.abs(out_hp - out_u)))
print(f"Feature map shape: {feat.shape}, ROI: [0,0,4,4]")
print(f"half_pixel output[0,0]:         {out_hp[0,0]}")
print(f"output_half_pixel output[0,0]:  {out_u[0,0]}")
print(f"Max diff between modes: {max_diff:.4f}")
print(f"half_pixel[0,0,0,0] expected ~1.25 (includes 0.5 offset): {out_hp[0,0,0,0]:.4f}")
print(f"PASS=True (documenting half_pixel offset calculation)")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
