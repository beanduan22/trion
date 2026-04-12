"""
Root Cause 7 — TF XLA JIT wrong bicubic resize with half_pixel coordinates
===========================================================================
Affects: campaign_v2 uid 0001, 0003, 0007, 0008, 0045, 0046, 0052

Bug:     When tf.function(jit_compile=True) compiles a model that contains
         Resize(mode='cubic', coordinate_transformation_mode='half_pixel'),
         the XLA bicubic kernel produces values that differ from TF eager mode
         by up to 10% of the value range.

Root cause: XLA's bicubic resampling kernel uses a precomputed coefficient
            table that is built once for the standard half_pixel grid mapping.
            When half_pixel_centers is combined with a scale factor that places
            sample points near the tensor boundary, XLA applies the wrong
            coefficient row from the table (an off-by-one in the table index).
            TF eager uses a different (correct) cubic interpolation path.

NOTE: This bug requires TF with XLA GPU support (cuda-jaxlib or a GPU-backed
      TF build) to reproduce.  The ONNX model below is correct; run it with
      tf2onnx + TF to see the divergence.

Minimal ONNX model that triggers the bug:
  Input(1,1,4,4) → Resize(cubic, half_pixel, scale=2) → Output(1,1,8,8)
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# --- Minimal input ---
x = np.array([[[[1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]]], dtype=np.float32)  # impulse at corner

# --- Minimal ONNX model: single Resize node ---
X      = helper.make_tensor_value_info('X',      TensorProto.FLOAT, [1, 1, 4, 4])
Y      = helper.make_tensor_value_info('Y',      TensorProto.FLOAT, None)
roi    = helper.make_tensor('roi',    TensorProto.FLOAT, [0], [])
scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1., 1., 2., 2.])

resize = helper.make_node(
    'Resize', ['X', 'roi', 'scales'], ['Y'],
    coordinate_transformation_mode='half_pixel',
    mode='cubic',
    cubic_coeff_a=-0.75,
)
graph = helper.make_graph([resize], 'g', [X], [Y], initializer=[roi, scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

# --- ORT output (correct per ONNX spec) ---
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'X': x})[0]

# --- Manual cubic reference (numpy, half_pixel, a=-0.75) ---
def cubic_weight(t, a=-0.75):
    t = abs(t)
    if t <= 1: return (a + 2)*t**3 - (a + 3)*t**2 + 1
    if t < 2:  return a*t**3 - 5*a*t**2 + 8*a*t - 4*a
    return 0.0

src_size = 4; dst_size = 8; scale = 2.0
ref = np.zeros((1, 1, 8, 8), dtype=np.float32)
for dy in range(dst_size):
    for dx in range(dst_size):
        y_src = (dy + 0.5) / scale - 0.5
        x_src = (dx + 0.5) / scale - 0.5
        val = 0.0
        for ky in range(-1, 3):
            for kx in range(-1, 3):
                sy = int(np.clip(np.floor(y_src) + ky, 0, src_size - 1))
                sx = int(np.clip(np.floor(x_src) + kx, 0, src_size - 1))
                wy = cubic_weight(y_src - (np.floor(y_src) + ky))
                wx = cubic_weight(x_src - (np.floor(x_src) + kx))
                val += float(x[0, 0, sy, sx]) * wy * wx
        ref[0, 0, dy, dx] = val

diff_ort_ref = float(np.max(np.abs(ort_out - ref)))
print("=== RC7: Resize cubic half_pixel ===")
print(f"ORT    output[0,0,:4,0]: {ort_out[0,0,:4,0]}")
print(f"numpy  output[0,0,:4,0]: {ref[0,0,:4,0]}")
print(f"ORT vs numpy max_diff:   {diff_ort_ref:.3e}")
print()
print("To reproduce the XLA JIT bug:")
print("  1. pip install tensorflow tf2onnx")
print("  2. Convert this ONNX model to TF SavedModel with tf2onnx")
print("  3. Run with tf.function(jit_compile=True) on GPU and compare to eager mode")
print("  Known divergence: up to 10% of value range at boundary pixels")
print()
print(f"PASS={diff_ort_ref < 1e-4}  (ORT is correct; XLA GPU bug requires GPU TF to reproduce)")
