"""
Root Cause 2 — Resize linear coordinate_transformation_mode mismatch
=====================================================================
Affects: campaign_v2 uid 0000, 0016, 0027, 0044, 0054 (ORT vs pytorch divergence)

Bug:     ONNX models specify Resize(coordinate_transformation_mode='asymmetric',
         mode='linear').  ORT implements this correctly per ONNX spec.
         onnx2torch uses PyTorch's default bilinear mode, which applies
         'pytorch_half_pixel' coordinates — a different formula that adds
         a half-pixel offset.

Root cause: onnx2torch silently maps ONNX coordinate modes to PyTorch
            defaults without honouring the model's explicit attribute.

         ONNX asymmetric formula:
           x_src = x_dst / scale               ← no offset
         pytorch_half_pixel formula:
           x_src = (x_dst + 0.5) / scale - 0.5 ← half-pixel offset

         For scale=2, a 3-element input:
           x_dst=1  →  asymmetric: 0.5     → lerp(0,1) = 2.0
                    →  half_pixel: 0.25    → lerp(0,1) = 1.0   ← wrong
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# --- Minimal input ---
# 3-element 1-D spatial array, scale 2x
x = np.array([[[[0., 4., 8.]]]], dtype=np.float32)   # shape [1,1,1,3]

# --- Minimal ONNX model: single Resize node ---
X      = helper.make_tensor_value_info('X',      TensorProto.FLOAT, [1, 1, 1, 3])
Y      = helper.make_tensor_value_info('Y',      TensorProto.FLOAT, None)
roi    = helper.make_tensor('roi',    TensorProto.FLOAT, [0], [])
scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1., 1., 1., 2.])

resize = helper.make_node(
    'Resize', ['X', 'roi', 'scales'], ['Y'],
    coordinate_transformation_mode='asymmetric',
    mode='linear',
)
graph = helper.make_graph([resize], 'g', [X], [Y], initializer=[roi, scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

# --- Run ORT (correct per spec) ---
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'X': x})[0].ravel()

# --- Manual spec reference: asymmetric linear ---
# x_src = x_dst / scale; linear interp between floor and ceil indices
vals = x.ravel()
def lerp(a, b, t): return a + (b - a) * t
spec_out = []
for j in range(6):
    src = j / 2.0
    lo, hi = int(np.floor(src)), int(np.ceil(src))
    lo = min(lo, 2); hi = min(hi, 2)
    spec_out.append(lerp(vals[lo], vals[hi], src - lo))
spec_out = np.array(spec_out, dtype=np.float32)

# --- What onnx2torch produces (pytorch_half_pixel, bilinear) ---
# x_src = (x_dst + 0.5) / scale - 0.5
wrong_out = []
for j in range(6):
    src = (j + 0.5) / 2.0 - 0.5
    lo, hi = int(np.floor(src)), int(np.ceil(src))
    lo = max(lo, 0); hi = min(hi, 2)
    wrong_out.append(lerp(vals[lo], vals[hi], src - lo))
wrong_out = np.array(wrong_out, dtype=np.float32)

print("Input:                ", x.ravel())
print("ORT output (correct): ", ort_out)
print("Spec reference:       ", spec_out)
print("onnx2torch output:    ", wrong_out)
print()
print(f"ORT matches spec:     {np.allclose(ort_out, spec_out, atol=1e-5)}")
print(f"Max diff ORT vs onnx2torch: {np.max(np.abs(ort_out - wrong_out)):.4f}")
print()
print(f"PASS={np.allclose(ort_out, spec_out, atol=1e-5)}")
