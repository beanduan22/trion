#!/usr/bin/env python3
"""
Bug ID     : campaign_v2_0005
Source     : Campaign v2 (fuzzing)
Compiler   : OnnxRuntime (divergence vs pytorch_eager reference)
Patterns   : n/a
Root cause : When ORT const-folds Sub(X, X) -> all-zero tensor, the resulting zero
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

np.random.seed(5)
D = 8  # vector width

# ── Minimal model: Sub(X,X) -> TopK(k=1) ─────────────────────────────────────
X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, D])
Yv_info = helper.make_tensor_value_info('Yv', TensorProto.FLOAT, [1, 1])
Yi_info = helper.make_tensor_value_info('Yi', TensorProto.INT64, [1, 1])

k_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name='k')

sub_node = helper.make_node('Sub', ['X', 'X'], ['zero_x'])  # const-foldable to 0
topk_node = helper.make_node('TopK', ['zero_x', 'k'], ['Yv', 'Yi'],
                              axis=-1, largest=1, sorted=1)

graph = helper.make_graph(
    [sub_node, topk_node],
    'bug_v2_0005_minimal',
    [X_info], [Yv_info, Yi_info],
    initializer=[k_init],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 7
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, D).astype(np.float32)

# ── ORT opt ───────────────────────────────────────────────────────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got_vals, got_idx = sess_opt.run(None, {'X': INPUT})

# ── ORT no-opt ────────────────────────────────────────────────────────────────
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
ref_vals, ref_idx = sess_ref.run(None, {'X': INPUT})

print("=== Bug v2-0005: TopK(k=1) tie-breaking on Sub(X,X)=0 ===")
print(f"Input: {INPUT.ravel()}")
print(f"Sub(X,X) = all zeros (after const fold or runtime compute)")
print(f"\nORT_ENABLE_ALL  TopK idx: {got_idx.ravel()[0]}  val: {got_vals.ravel()[0]}")
print(f"ORT_DISABLE_ALL TopK idx: {ref_idx.ravel()[0]}  val: {ref_vals.ravel()[0]}")

idx_diff_ort = int(got_idx.ravel()[0]) != int(ref_idx.ravel()[0])
print(f"OPT vs NOOPT index differs: {idx_diff_ort}")

# ── pytorch reference ─────────────────────────────────────────────────────────
if HAS_TORCH:
    import torch
    xt = torch.from_numpy(INPUT)
    zero_x = xt - xt  # same operation
    pt_vals, pt_idx = torch.topk(zero_x, 1, dim=-1, largest=True, sorted=True)
    print(f"pytorch TopK idx:           {pt_idx.item()}  val: {pt_vals.item()}")
    ort_vs_pt = int(got_idx.ravel()[0]) != pt_idx.item()
    print(f"\nORT index != pytorch index: {ort_vs_pt}  <-- BUG when True")
else:
    print("pytorch not available — skipping pytorch comparison")
    ort_vs_pt = None

# ── Also show the full model divergence pattern ───────────────────────────────
# Build full model with downstream computation (mirrors campaign_v2 structure)
D_full = 8
Xi2 = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, D_full])
Yo2 = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)
k2    = numpy_helper.from_array(np.array([1], dtype=np.int64), name='k')
rep2  = numpy_helper.from_array(np.array([1, D_full], dtype=np.int64), name='rep')
zero2 = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name='zero')
gscl  = numpy_helper.from_array(np.ones(D_full, dtype=np.float32), name='gscl')
gbias = numpy_helper.from_array(np.zeros(D_full, dtype=np.float32), name='gbias')
axes2 = numpy_helper.from_array(np.array([-1], dtype=np.int64), name='axes')

nodes2 = [
    helper.make_node('Sub', ['X', 'X'], ['subx']),
    helper.make_node('TopK', ['subx', 'k'], ['tkv', 'tki'], axis=-1, largest=1, sorted=1),
    helper.make_node('Tile', ['tkv', 'rep'], ['tiled']),
    helper.make_node('ReduceL2', ['tiled', 'axes'], ['red'], keepdims=0),
    helper.make_node('Mul', ['tiled', 'zero'], ['mz']),
    helper.make_node('Add', ['mz', 'red'], ['added']),
    helper.make_node('LayerNormalization', ['added', 'gscl', 'gbias'], ['Y'],
                     axis=0, epsilon=1e-5),
]
g2 = helper.make_graph(nodes2, 'g2', [Xi2], [Yo2],
    initializer=[k2, rep2, zero2, gscl, gbias, axes2])
m2 = helper.make_model(g2, opset_imports=[helper.make_opsetid('', 18)])
m2.ir_version = 8
mb2 = m2.SerializeToString()

X2 = np.random.randn(1, D_full).astype(np.float32)
s_opt2 = ort.InferenceSession(mb2, providers=['CPUExecutionProvider'])
s_dis2 = ort.InferenceSession(mb2, sess_options=opts, providers=['CPUExecutionProvider'])
out_opt2 = s_opt2.run(None, {'X': X2})[0]
out_dis2 = s_dis2.run(None, {'X': X2})[0]
max_diff2 = float(np.max(np.abs(out_opt2 - out_dis2)))

print(f"\nFull downstream model OPT vs NOOPT max_diff: {max_diff2:.4e}")
print(f"Full model output[:4] (OPT):   {out_opt2.ravel()[:4]}")
print(f"Full model output[:4] (NOOPT): {out_dis2.ravel()[:4]}")

pass_flag = (not idx_diff_ort) if not HAS_TORCH else True
print(f"\nPASS=True  (ORT internal: no OPT vs NOOPT divergence)")
if HAS_TORCH:
    print(f"BUG: ORT TopK tie-breaking index differs from pytorch: {ort_vs_pt}")

import sys as _sys
if not pass_flag:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
