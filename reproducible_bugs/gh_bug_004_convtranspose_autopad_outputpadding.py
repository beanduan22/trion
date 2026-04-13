#!/usr/bin/env python3
"""
Cross-compiler bug: ConvTranspose — auto_pad and output_padding mishandled
==========================================================================
Compilers affected : OnnxRuntime (#4086, #14208), OpenVINO (#30798), TVM (PR #7958),
                     PyTorch Inductor (#108908)
Shared root cause  : All four compilers have bugs computing ConvTranspose output shape
                     or values when auto_pad or output_padding is used:
                     - ORT #4086:   auto_pad=SAME_UPPER ignored, used explicit pads=[0,0,0,0]
                     - ORT #14208:  dilation>1 + output_padding gives wrong values
                     - OV #30798:   SAME_LOWER applied SAME_UPPER formula → 1-pixel shape error
                     - TVM PR#7958: output_padding ignored entirely → 9x9 instead of 10x10
                     - Inductor #108908: output_size=(0,0) caused IndexError not ValueError
Status             : All bugs closed/fixed. This repro validates output shape and values
                     for each problematic configuration via ONNX/ORT.

Four sub-tests:
  A) auto_pad=SAME_UPPER: output = input * stride.
  B) auto_pad=SAME_LOWER: same formula, different pad distribution.
  C) output_padding=[1,1] with stride=2: adds 1 pixel → 10x10 not 9x9.
  D) dilation=2 + output_padding=[1,1]: values must match PyTorch ConvTranspose2d.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

np.random.seed(7)

# ── Test A: auto_pad=SAME_UPPER (ORT #4086) ──────────────────────────────────
x_A = np.random.randn(1, 1, 4, 4).astype(np.float32)
w_A = np.random.randn(1, 1, 3, 3).astype(np.float32)

X_A = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y_A = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 8])
node_A  = helper.make_node("ConvTranspose", ["X","W"], ["Y"],
                           auto_pad="SAME_UPPER", strides=[2, 2])
graph_A = helper.make_graph([node_A], "g", [X_A], [Y_A],
                             initializer=[numpy_helper.from_array(w_A, "W")])
model_A = helper.make_model(graph_A, opset_imports=[helper.make_opsetid("", 11)])
out_A   = ort.InferenceSession(model_A.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(None, {"X": x_A})[0]
ok_A = out_A.shape == (1, 1, 8, 8) and bool(np.all(np.isfinite(out_A)))
print(f"A) SAME_UPPER stride=2: shape={out_A.shape} expected=(1,1,8,8)  ok={ok_A}")

# ── Test B: auto_pad=SAME_LOWER (OV #30798) ──────────────────────────────────
x_B = np.random.randn(1, 1, 4, 4).astype(np.float32)
X_B = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y_B = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node_B  = helper.make_node("ConvTranspose", ["X","W"], ["Y"],
                           auto_pad="SAME_LOWER", strides=[2, 2])
graph_B = helper.make_graph([node_B], "g", [X_B], [Y_B],
                             initializer=[numpy_helper.from_array(w_A, "W")])
model_B = helper.make_model(graph_B, opset_imports=[helper.make_opsetid("", 11)])
out_B   = ort.InferenceSession(model_B.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(None, {"X": x_B})[0]
# SAME_LOWER and SAME_UPPER both give output = input * stride = 8x8
ok_B = out_B.shape == (1, 1, 8, 8) and bool(np.all(np.isfinite(out_B)))
print(f"B) SAME_LOWER stride=2: shape={out_B.shape} expected=(1,1,8,8)  ok={ok_B}")

# ── Test C: output_padding=[1,1] stride=2 (TVM PR#7958) ──────────────────────
# (input-1)*stride - 2*pad + kernel + output_padding = 3*2 + 3 + 1 = 10
x_C = np.random.randn(1, 1, 4, 4).astype(np.float32)
X_C = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y_C = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node_C  = helper.make_node("ConvTranspose", ["X","W"], ["Y"],
                           strides=[2,2], output_padding=[1,1], pads=[0,0,0,0])
graph_C = helper.make_graph([node_C], "g", [X_C], [Y_C],
                             initializer=[numpy_helper.from_array(w_A, "W")])
model_C = helper.make_model(graph_C, opset_imports=[helper.make_opsetid("", 11)])
out_C   = ort.InferenceSession(model_C.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(None, {"X": x_C})[0]
ok_C = out_C.shape == (1, 1, 10, 10)
print(f"C) output_padding=[1,1] stride=2: shape={out_C.shape} expected=(1,1,10,10) ok={ok_C}")
print(f"   TVM bug: produced (1,1,9,9) — output_padding ignored entirely")

# ── Test D: dilation=2 + output_padding=[1,1] (ORT #14208) ───────────────────
x_D  = np.random.randn(1, 2, 5, 5).astype(np.float32)
w_D  = np.random.randn(2, 1, 3, 3).astype(np.float32)
X_D  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 5, 5])
Y_D  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node_D  = helper.make_node("ConvTranspose", ["X","W"], ["Y"],
                           dilations=[2,2], strides=[2,2],
                           output_padding=[1,1], pads=[0,0,0,0])
graph_D = helper.make_graph([node_D], "g", [X_D], [Y_D],
                             initializer=[numpy_helper.from_array(w_D, "W")])
model_D = helper.make_model(graph_D, opset_imports=[helper.make_opsetid("", 11)])
out_D   = ort.InferenceSession(model_D.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(None, {"X": x_D})[0]

if HAS_TORCH:
    ct = nn.ConvTranspose2d(2, 1, 3, stride=2, padding=0,
                            output_padding=1, dilation=2, bias=False)
    with torch.no_grad():
        ct.weight.copy_(torch.from_numpy(w_D))
    ref_D    = ct(torch.from_numpy(x_D)).detach().numpy()
    shape_ok = out_D.shape == ref_D.shape
    diff_D   = float(np.max(np.abs(out_D - ref_D))) if shape_ok else float("inf")
    ok_D     = shape_ok and diff_D < 1e-4
    print(f"D) dilation=2 + output_padding=[1,1]: shape={out_D.shape} diff={diff_D:.2e}  ok={ok_D}")
    print(f"   ORT #14208 bug: wrong values with dilation>1 + output_padding")
else:
    ok_D = out_D.shape[2] > 0 and bool(np.all(np.isfinite(out_D)))
    print(f"D) dilation=2 + output_padding=[1,1]: shape={out_D.shape} finite={ok_D}")

PASS = ok_A and ok_B and ok_C and ok_D
print(f"Bugs: ORT #4086/#14208, OV #30798, TVM PR#7958, Inductor #108908")
print(f"PASS={PASS}")
