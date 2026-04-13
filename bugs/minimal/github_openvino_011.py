#!/usr/bin/env python3
"""
Bug ID     : github_openvino_011
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : avgpool cpu int8 ceil
Root cause : OpenVINO Bug #20815 - AvgPool CPU int8 with ceil_mode gives 71.4% wrong outputs
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO Bug #20815 - AvgPool CPU int8 with ceil_mode gives 71.4% wrong outputs
# https://github.com/openvinotoolkit/openvino/issues/20815
# OV bug: int8 kernel for AvgPool ceil_mode=True divided border windows by full kernel
# size instead of actual overlap count (e.g., divided 2-elem border by 3 instead of 2)
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Input [1,1,1,6], kernel=3, stride=2, ceil_mode=1
# Windows: [0:3]->4.0, [2:5]->8.0, [4:6](border,2 elem)->11.0
x = np.array([[[[2., 4., 6., 8., 10., 12.]]]], dtype=np.float32)

X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 6])
Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("AveragePool", ["X"], ["Y"],
                        kernel_shape=[1, 3], strides=[1, 2],
                        ceil_mode=1, count_include_pad=0)
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"X": x})[0]

# Border window [4:6] has 2 elements -> correct avg = (10+12)/2 = 11.0
# OV int8 bug: divided by 3 (full kernel) -> (10+12)/3 ≈ 7.33
expected = np.array([[[[4., 8., 11.]]]], dtype=np.float32)
max_diff  = float(np.max(np.abs(out - expected)))
print(f"input: {x[0,0,0,:]}")
print(f"ort_out:  {out[0,0,0,:]}")
print(f"expected: {expected[0,0,0,:]}")
print(f"border window: ort={out[0,0,0,2]:.4f}  correct=11.0  wrong(OV bug)=7.33")
print(f"max_diff: {max_diff:.2e}")
print(f"PASS={max_diff < 1e-4}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
