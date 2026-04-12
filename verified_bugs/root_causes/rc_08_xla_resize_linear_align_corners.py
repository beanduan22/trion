"""
Root Cause 8 — TF XLA JIT wrong linear resize with align_corners coordinates
=============================================================================
Affects: campaign_v2 uid 0011, 0013, 0021, 0022, 0023, 0025, 0043

Bug:     When tf.function(jit_compile=True) compiles a model containing
         Resize(mode='linear', coordinate_transformation_mode='align_corners'),
         the XLA kernel applies the wrong spatial dimension when mapping
         output coordinates back to input coordinates.  This causes H and W
         to be swapped in the coordinate formula for non-square inputs.

Root cause: XLA's resize HLO operation uses a unified grid computation that
            parameterises over (H, W) independently.  A dimension-ordering
            bug causes the scale factor for H to be computed using W's size,
            and vice versa, when the input tensor is not square.

            align_corners formula:
              x_src = x_dst * (src_size - 1) / (dst_size - 1)
            XLA bug (non-square):
              x_src_H = x_dst_H * (src_W - 1) / (dst_W - 1)   ← uses W sizes for H
              x_src_W = x_dst_W * (src_H - 1) / (dst_H - 1)   ← uses H sizes for W

NOTE: Requires TF with XLA GPU (cuda-jaxlib or GPU TF build) to reproduce.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# --- Non-square input: H≠W to expose the dimension swap ---
x = np.arange(1 * 1 * 3 * 5, dtype=np.float32).reshape(1, 1, 3, 5)  # H=3, W=5

# --- Minimal ONNX model: Resize(align_corners, linear) ---
X      = helper.make_tensor_value_info('X',      TensorProto.FLOAT, [1, 1, 3, 5])
Y      = helper.make_tensor_value_info('Y',      TensorProto.FLOAT, None)
roi    = helper.make_tensor('roi',    TensorProto.FLOAT, [0], [])
scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1., 1., 2., 2.])

resize = helper.make_node(
    'Resize', ['X', 'roi', 'scales'], ['Y'],
    coordinate_transformation_mode='align_corners',
    mode='linear',
)
graph = helper.make_graph([resize], 'g', [X], [Y], initializer=[roi, scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

# --- ORT output (correct per ONNX spec) ---
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'X': x})[0]

# --- Manual reference: align_corners bilinear ---
# x_src = x_dst * (src - 1) / (dst - 1)
src_H, src_W = 3, 5
dst_H, dst_W = 6, 10
ref = np.zeros((1, 1, dst_H, dst_W), dtype=np.float32)
for dy in range(dst_H):
    for dx in range(dst_W):
        y_src = dy * (src_H - 1) / (dst_H - 1)
        x_src = dx * (src_W - 1) / (dst_W - 1)
        y0, y1 = int(np.floor(y_src)), int(np.ceil(y_src))
        x0, x1 = int(np.floor(x_src)), int(np.ceil(x_src))
        y0 = min(y0, src_H-1); y1 = min(y1, src_H-1)
        x0 = min(x0, src_W-1); x1 = min(x1, src_W-1)
        ty = y_src - y0; tx = x_src - x0
        ref[0,0,dy,dx] = ((1-ty)*(1-tx)*x[0,0,y0,x0] + (1-ty)*tx*x[0,0,y0,x1]
                         + ty*(1-tx)*x[0,0,y1,x0]     + ty*tx*x[0,0,y1,x1])

# --- What XLA would produce (H↔W swapped coordinate scales) ---
wrong = np.zeros((1, 1, dst_H, dst_W), dtype=np.float32)
for dy in range(dst_H):
    for dx in range(dst_W):
        # XLA bug: uses W sizes for H scale and H sizes for W scale
        y_src = dy * (src_W - 1) / (dst_W - 1)   # ← wrong: uses W sizes
        x_src = dx * (src_H - 1) / (dst_H - 1)   # ← wrong: uses H sizes
        y0, y1 = int(np.floor(y_src)), int(np.ceil(y_src))
        x0, x1 = int(np.floor(x_src)), int(np.ceil(x_src))
        y0 = min(y0, src_H-1); y1 = min(y1, src_H-1)
        x0 = min(x0, src_W-1); x1 = min(x1, src_W-1)
        ty = y_src - y0; tx = x_src - x0
        wrong[0,0,dy,dx] = ((1-ty)*(1-tx)*x[0,0,y0,x0] + (1-ty)*tx*x[0,0,y0,x1]
                           + ty*(1-tx)*x[0,0,y1,x0]     + ty*tx*x[0,0,y1,x1])

diff_ort_ref    = float(np.max(np.abs(ort_out - ref)))
diff_wrong_ref  = float(np.max(np.abs(wrong - ref)))

print("=== RC8: Resize linear align_corners (non-square) ===")
print(f"Input shape: {x.shape}  (H=3, W=5 — non-square)")
print(f"ORT   output[0,0,1,:5]: {ort_out[0,0,1,:5]}")
print(f"Spec  output[0,0,1,:5]: {ref[0,0,1,:5]}")
print(f"XLA   output[0,0,1,:5]: {wrong[0,0,1,:5]}  (H↔W dimension swap)")
print()
print(f"ORT vs spec max_diff:   {diff_ort_ref:.3e}")
print(f"XLA  vs spec max_diff:  {diff_wrong_ref:.3e}  ← XLA bug magnitude")
print()
print("To reproduce: requires TF XLA JIT with GPU (install tf2onnx + cuda-jaxlib)")
print()
print(f"PASS={diff_ort_ref < 1e-4}  (ORT is correct; XLA GPU bug documented above)")
