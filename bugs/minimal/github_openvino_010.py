#!/usr/bin/env python3
"""
Bug ID     : github_openvino_010
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : depth to space blocks first mode
Root cause : OpenVINO Bug #29029 - DepthToSpace fails in blocks_first (DCR) mode on NPU
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO Bug #29029 - DepthToSpace fails in blocks_first (DCR) mode on NPU
# https://github.com/openvinotoolkit/openvino/issues/29029
# OV bug: NPU plugin threw error for DepthToSpace with mode=DCR; CPU/GPU worked fine
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Small reproducer: [1,16,2,2] with blocksize=2 -> [1,4,4,4]
bs = 2
C, H, W = 4, 2, 2
x = np.arange(C * bs * bs * H * W, dtype=np.float32).reshape(1, C * bs * bs, H, W)

X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, C * bs * bs, H, W])
Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, C, H * bs, W * bs])
node = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=bs, mode="DCR")
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"X": x})[0]

# DCR: reshape to [1, bs, bs, C, H, W], transpose, reshape
x_r    = x.reshape(1, bs, bs, C, H, W)
x_t    = x_r.transpose(0, 3, 4, 1, 5, 2)
expected = x_t.reshape(1, C, H * bs, W * bs)

max_diff = float(np.max(np.abs(out - expected)))
print(f"input shape: {x.shape}, output shape: {out.shape}")
print(f"ort_out[0,0,0,:4]:  {out[0,0,0,:4]}")
print(f"expected[0,0,0,:4]: {expected[0,0,0,:4]}")
print(f"max_diff: {max_diff:.2e}")
print(f"OV bug: NPU compilation failed for DCR mode (CPU was correct)")
print(f"PASS={max_diff < 1e-5}")

PASS = max_diff < 1e-5
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
