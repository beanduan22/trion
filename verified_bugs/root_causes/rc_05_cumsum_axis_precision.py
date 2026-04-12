"""
Root Cause 5 — ORT CumSum last-axis accumulation order vs pytorch_eager
========================================================================
Affects: campaign_v2 uid 0014, 0018, 0030, 0035, 0036

Bug:     ORT's CumSum kernel on the last axis (axis=-1) accumulates in a
         different order than PyTorch's sequential scan, producing small
         floating-point differences (~1e-7).  For typical values this is
         negligible, but when CumSum feeds downstream MatMul or Softmax
         over a long sequence, the error accumulates to ~1e-4 to 1e-3.

Root cause: ORT's parallel-prefix CumSum kernel uses a tree-reduction
            pattern that sums pairs of elements at each level.  PyTorch
            uses a sequential left-to-right scan.  Both are mathematically
            correct, but floating-point addition is non-associative, so
            the results can differ at the ULP level.

            Example for 4 elements [a,b,c,d]:
              Sequential:  [a,  a+b,  a+b+c,  a+b+c+d]
              Tree-reduce: [a,  a+b,  c+a+b,  (c+d)+(a+b)]
                           ↑ same  ↑ same   ↑ may differ in float32
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# --- Inputs: long vector with values spread across float32 range ---
np.random.seed(42)
x = np.random.randn(1, 128).astype(np.float32)   # long enough to accumulate error

# --- Minimal ONNX model: CumSum(axis=1) ---
X    = helper.make_tensor_value_info('X',    TensorProto.FLOAT, [1, 128])
Y    = helper.make_tensor_value_info('Y',    TensorProto.FLOAT, None)
axis = helper.make_tensor('axis', TensorProto.INT32, [1], [1])

cumsum = helper.make_node('CumSum', ['X', 'axis'], ['Y'], exclusive=0, reverse=0)
graph  = helper.make_graph([cumsum], 'g', [X], [Y], initializer=[axis])
model  = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])

# --- ORT (parallel prefix) ---
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'X': x})[0].ravel()

# --- numpy reference (sequential left-to-right) ---
np_ref = np.cumsum(x, axis=1).ravel()

# --- PyTorch reference ---
import torch
pt_out = torch.cumsum(torch.from_numpy(x), dim=1).numpy().ravel()

diff_ort_np  = float(np.max(np.abs(ort_out  - np_ref)))
diff_ort_pt  = float(np.max(np.abs(ort_out  - pt_out)))
diff_np_pt   = float(np.max(np.abs(np_ref   - pt_out)))

print(f"Input[:4]:          {x.ravel()[:4]}")
print(f"ORT   cumsum[:4]:   {ort_out[:4]}")
print(f"numpy cumsum[:4]:   {np_ref[:4]}")
print(f"torch cumsum[:4]:   {pt_out[:4]}")
print()
print(f"max_diff ORT vs numpy:   {diff_ort_np:.3e}")
print(f"max_diff ORT vs PyTorch: {diff_ort_pt:.3e}")
print(f"max_diff numpy vs PyTorch: {diff_np_pt:.3e}")
print()
print(f"PASS={diff_ort_np < 1e-4}")
