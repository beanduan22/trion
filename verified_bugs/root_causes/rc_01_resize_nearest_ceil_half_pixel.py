"""
Root Cause 1 — Resize nearest_mode=ceil not supported by onnx2torch
====================================================================
Affects: campaign_v2 uid 0002, 0004, 0010, 0015, 0019, 0024, 0028, 0029,
         0031, 0037, 0039, 0041, 0053, 0055 (all showing ORT vs pytorch divergence)

Bug:     ONNX models specify Resize(coordinate_transformation_mode='half_pixel',
         mode='nearest', nearest_mode='ceil').
         ORT implements this correctly per the ONNX spec.
         onnx2torch (used as oracle in Trion) silently falls back to
         nearest_mode='floor' with coordinate_transformation_mode='asymmetric',
         producing different pixel selections.

Root cause: onnx2torch limitation — it warns but does not raise an error,
            making the oracle silently incorrect.

         ONNX spec formula for half_pixel + ceil:
           x_src = ceil((x_dst + 0.5) / scale - 0.5)
         onnx2torch actually computes:
           x_src = floor(x_dst / scale)     ← wrong mode, wrong rounding
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# --- Minimal input ---
# 1-D spatial array, scale factor 2x
x = np.array([[[[1., 2., 3., 4.]]]], dtype=np.float32)   # shape [1,1,1,4]

# --- Minimal ONNX model: single Resize node ---
X     = helper.make_tensor_value_info('X',      TensorProto.FLOAT, [1, 1, 1, 4])
Y     = helper.make_tensor_value_info('Y',      TensorProto.FLOAT, None)
roi   = helper.make_tensor('roi',    TensorProto.FLOAT, [0], [])
scales= helper.make_tensor('scales', TensorProto.FLOAT, [4], [1., 1., 1., 2.])

resize = helper.make_node(
    'Resize', ['X', 'roi', 'scales'], ['Y'],
    coordinate_transformation_mode='half_pixel',
    mode='nearest',
    nearest_mode='ceil',
)
graph = helper.make_graph([resize], 'g', [X], [Y], initializer=[roi, scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

# --- Run ORT (correct per spec) ---
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'X': x})[0].ravel()

# --- Manual spec reference: half_pixel + ceil ---
# x_src = ceil((x_dst + 0.5) / scale - 0.5), clipped to [0, src_size-1]
src_size = 4
dst_size = 8
scale    = 2.0
spec_out = np.array([x.ravel()[min(max(int(np.ceil((j + 0.5) / scale - 0.5)), 0), src_size - 1)]
                     for j in range(dst_size)], dtype=np.float32)

# --- What onnx2torch produces (floor + asymmetric fallback) ---
# x_src = floor(x_dst / scale), clipped to [0, src_size-1]
wrong_out = np.array([x.ravel()[min(int(np.floor(j / scale)), src_size - 1)]
                      for j in range(dst_size)], dtype=np.float32)

print("Input:                ", x.ravel())
print("ORT output (correct): ", ort_out)
print("Spec reference:       ", spec_out)
print("onnx2torch output:    ", wrong_out)
print()
print(f"ORT matches spec:     {np.allclose(ort_out, spec_out)}")
print(f"ORT vs onnx2torch differ at indices: {np.where(ort_out != wrong_out)[0].tolist()}")
print()
print(f"PASS={np.allclose(ort_out, spec_out)}")
