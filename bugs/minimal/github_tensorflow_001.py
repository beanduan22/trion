#!/usr/bin/env python3
"""
Bug ID     : github_tensorflow_001
Source     : GitHub — TensorFlow XLA JIT
Compiler   : TensorFlow XLA JIT
Patterns   : resize bilinear aligncorners tflite
Root cause : Bug: TFLite converter silently forced align_corners=False (half_pixel) instead of True (asymmetric).
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: TFLite converter silently forced align_corners=False (half_pixel) instead of True (asymmetric).
np.random.seed(17)
src = np.random.rand(1, 3, 4, 4).astype(np.float32)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1., 1., 2., 2.])

n_half = helper.make_node("Resize", ["X","","scales"], ["Y"],
    mode="linear", coordinate_transformation_mode="half_pixel")
g_half = helper.make_graph([n_half], "g", [X], [Y], initializer=[scales])
m_half = helper.make_model(g_half, opset_imports=[helper.make_opsetid("", 13)])
sess_half = ort.InferenceSession(m_half.SerializeToString(), providers=["CPUExecutionProvider"])

n_asym = helper.make_node("Resize", ["X","","scales"], ["Y"],
    mode="linear", coordinate_transformation_mode="asymmetric")
g_asym = helper.make_graph([n_asym], "g", [X], [Y], initializer=[scales])
m_asym = helper.make_model(g_asym, opset_imports=[helper.make_opsetid("", 13)])
sess_asym = ort.InferenceSession(m_asym.SerializeToString(), providers=["CPUExecutionProvider"])

out_half = sess_half.run(None, {"X": src})[0]
out_asym = sess_asym.run(None, {"X": src})[0]

max_diff = float(np.max(np.abs(out_half - out_asym)))
print(f"half_pixel (TFLite default) output[0,0,0,:4]: {out_half[0,0,0,:4]}")
print(f"asymmetric (align_corners≈True) [0,0,0,:4]:  {out_asym[0,0,0,:4]}")
print(f"Max abs diff: {max_diff:.6f}")
print(f"TF bug: converter silently changed align_corners=True -> False (half_pixel)")
print(f"Bug reproduced (outputs differ): {max_diff > 1e-4}")
PASS = max_diff > 1e-4  # confirms the two modes differ (the silent switch causes error)
print(f"PASS={PASS}")

import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
