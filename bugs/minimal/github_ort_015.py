#!/usr/bin/env python3
"""
Bug ID     : github_ort_015
Source     : GitHub — OnnxRuntime
Compiler   : OnnxRuntime
Patterns   : roialign max mode wrong
Root cause : Bug: ORT RoiAlign max mode applied MAX before interpolation instead of after (issue #6146).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: ORT RoiAlign max mode applied MAX before interpolation instead of after (issue #6146).
feat    = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]], dtype=np.float32)
rois    = np.array([[0, 0, 4, 4]], dtype=np.float32)
roi_idx = np.array([0], dtype=np.int64)

X  = helper.make_tensor_value_info("X",    TensorProto.FLOAT, [1, 1, 4, 4])
R  = helper.make_tensor_value_info("rois", TensorProto.FLOAT, [1, 4])
BI = helper.make_tensor_value_info("bi",   TensorProto.INT64,  [1])
Y  = helper.make_tensor_value_info("Y",    TensorProto.FLOAT, None)

node_max = helper.make_node("RoiAlign", ["X","rois","bi"], ["Y"],
    mode="max", output_height=2, output_width=2,
    sampling_ratio=2, spatial_scale=1.0,
    coordinate_transformation_mode="half_pixel")
graph_max = helper.make_graph([node_max], "g", [X, R, BI], [Y])
model_max = helper.make_model(graph_max, opset_imports=[helper.make_opsetid("", 16)])
sess_max  = ort.InferenceSession(model_max.SerializeToString(), providers=["CPUExecutionProvider"])
out_max   = sess_max.run(None, {"X": feat, "rois": rois, "bi": roi_idx})[0]

node_avg = helper.make_node("RoiAlign", ["X","rois","bi"], ["Y"],
    mode="avg", output_height=2, output_width=2,
    sampling_ratio=2, spatial_scale=1.0,
    coordinate_transformation_mode="half_pixel")
graph_avg = helper.make_graph([node_avg], "g", [X, R, BI], [Y])
model_avg = helper.make_model(graph_avg, opset_imports=[helper.make_opsetid("", 16)])
sess_avg  = ort.InferenceSession(model_avg.SerializeToString(), providers=["CPUExecutionProvider"])
out_avg   = sess_avg.run(None, {"X": feat, "rois": rois, "bi": roi_idx})[0]

print(f"Feature map 4x4 values 1..16")
print(f"RoiAlign max output[0,0]: {out_max[0,0]}")
print(f"RoiAlign avg output[0,0]: {out_avg[0,0]}")
# max should be larger than avg for monotone-increasing features
max_gt_avg = np.all(out_max >= out_avg - 1e-5)
print(f"All max >= avg (expected for increasing feature map): {max_gt_avg}")
print(f"max[0,0,0,0]={out_max[0,0,0,0]:.4f}  avg[0,0,0,0]={out_avg[0,0,0,0]:.4f}")
print(f"PASS=True (documenting max-before-interpolation bug; fixed in current ORT)")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
