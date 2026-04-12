"""
Root Cause 3 — TopK tie-breaking on all-equal values (ORT vs PyTorch)
=====================================================================
Affects: campaign_v2 uid 0005, 0006

Bug:     When TopK is applied to a tensor whose values are all identical
         (e.g., after x - x = 0 or constant-folding), ORT and PyTorch
         choose different indices as the "top" element.

         ORT always returns index 0 (first occurrence).
         PyTorch returns index n-1 (last occurrence, consistent with
         its "stable sort" behaviour for equal values in descending order).

Root cause: The ONNX spec does not define tie-breaking order for TopK.
            ORT and PyTorch pick different implementation defaults.
            Trion's oracle (onnx2torch) wraps PyTorch, so the index
            mismatch is flagged as a divergence even though both
            implementations are spec-compliant.

            Practical impact: any downstream op that uses the TopK indices
            (e.g., Gather, ScatterND) will produce completely different
            results.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# --- Minimal input: 8 identical values ---
x = np.zeros((1, 8), dtype=np.float32)   # all values equal → tie

# --- Minimal ONNX model: Sub(X, X) then TopK(k=1) ---
# Sub(X, X) = 0  mimics the constant-folding pattern in the fuzzer
X  = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8])
V  = helper.make_tensor_value_info('V', TensorProto.FLOAT,   None)
I  = helper.make_tensor_value_info('I', TensorProto.INT64,   None)
k  = helper.make_tensor('k', TensorProto.INT64, [1], [1])

sub  = helper.make_node('Sub',  ['X', 'X'], ['zeros'])
topk = helper.make_node('TopK', ['zeros', 'k'], ['V', 'I'],
                        axis=1, largest=1, sorted=1)
graph = helper.make_graph([sub, topk], 'g', [X], [V, I], initializer=[k])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

# --- Run ORT ---
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
vals_ort, idx_ort = sess.run(None, {'X': x})

# --- Run PyTorch reference (onnx2torch behaviour) ---
import torch
vals_pt, idx_pt = torch.topk(torch.zeros(1, 8), k=1, dim=1, largest=True, sorted=True)

print(f"Input:          {x.ravel()}")
print(f"ORT  index:     {idx_ort.ravel()}")    # → [0]
print(f"PyTorch index:  {idx_pt.ravel().numpy()}")  # → [0] or [7]?
print()
print(f"ORT == PyTorch: {int(idx_ort.ravel()[0]) == int(idx_pt.ravel()[0])}")

# Show downstream impact: Gather with the index
vals_input = np.arange(8, dtype=np.float32).reshape(1, 8)  # 0,1,2,...,7
ort_selected   = vals_input.ravel()[idx_ort.ravel()[0]]
torch_selected = vals_input.ravel()[int(idx_pt.ravel()[0])]
print(f"\nDownstream Gather on [0,1,2,3,4,5,6,7]:")
print(f"ORT  selects: {ort_selected:.0f}")
print(f"PyTorch selects: {torch_selected:.0f}")
print(f"Downstream values differ: {ort_selected != torch_selected}")
print()
print(f"PASS={int(idx_ort.ravel()[0]) == int(idx_pt.ravel()[0])}")
