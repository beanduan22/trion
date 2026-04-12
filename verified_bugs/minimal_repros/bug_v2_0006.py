#!/usr/bin/env python3
"""
Bug v2-0006 — ORT TopK(k=1) wrong result after Gather + LayerNorm fusion
Compiler  : OnnxRuntime (divergence vs pytorch_eager; rel-L2 ~5.4e2)
Root cause: ORT optimizer fuses Gather(embedding) + Add(input) + LayerNormalization
            into a GatherElementsWithEmbedding kernel.  The fused kernel may select
            a different TopK index on the LayerNorm output due to numerical differences
            in the fused accumulation order vs the sequential Gather->Add->LN path.
            The TopK index error then propagates through the remaining Tile+LN+Sigmoid
            chain, causing large cumulative divergence.
Status    : Active (ORT_DISABLE_ALL == ORT_ENABLE_ALL; divergence is vs pytorch_eager)

Minimal trigger: Gather(embeddings, idx=0) + Add(input) -> LayerNorm -> TopK(k=1)
  The fused GatherLayerNorm produces slightly different output than the sequential path,
  causing TopK to select a different index when values are very close.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(6)

# ── Model parameters ──────────────────────────────────────────────────────────
D = 16    # embedding/feature dimension (small for speed)
N_EMB = 4 # number of embedding rows

# ── Initializers ──────────────────────────────────────────────────────────────
np.random.seed(6)
embeddings = np.random.randn(N_EMB, D).astype(np.float32) * 0.01
gln_idx    = np.array([0], dtype=np.int64)
gln_scale  = np.ones(D, dtype=np.float32)
gln_bias   = np.zeros(D, dtype=np.float32)
tk_k       = np.array([1], dtype=np.int64)
tk_rep     = np.array([1, D], dtype=np.int64)
ln2_scale  = np.ones(D, dtype=np.float32)
ln2_bias   = np.zeros(D, dtype=np.float32)

# ── Build ONNX model ──────────────────────────────────────────────────────────
# Pattern: Gather(embed, idx) + Add(input) -> LayerNorm -> TopK(k=1) -> Tile -> LayerNorm
X_info = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, D])
Y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

gather_node = helper.make_node('Gather', ['embeddings', 'gln_idx'], ['gathered'], axis=0)
add_node    = helper.make_node('Add',    ['gathered', 'X'],         ['added'])
ln1_node    = helper.make_node('LayerNormalization',
                                ['added', 'gln_scale', 'gln_bias'], ['ln1_out'],
                                axis=0, epsilon=1e-5)
topk_node   = helper.make_node('TopK', ['ln1_out', 'tk_k'], ['tk_vals', 'tk_idx'],
                                axis=-1, largest=1, sorted=1)
tile_node   = helper.make_node('Tile', ['tk_vals', 'tk_rep'], ['tiled'])
ln2_node    = helper.make_node('LayerNormalization',
                                ['tiled', 'ln2_scale', 'ln2_bias'], ['Y'],
                                axis=0, epsilon=1e-5)

graph = helper.make_graph(
    [gather_node, add_node, ln1_node, topk_node, tile_node, ln2_node],
    'bug_v2_0006',
    [X_info], [Y_info],
    initializer=[
        numpy_helper.from_array(embeddings, 'embeddings'),
        numpy_helper.from_array(gln_idx,    'gln_idx'),
        numpy_helper.from_array(gln_scale,  'gln_scale'),
        numpy_helper.from_array(gln_bias,   'gln_bias'),
        numpy_helper.from_array(tk_k,       'tk_k'),
        numpy_helper.from_array(tk_rep,     'tk_rep'),
        numpy_helper.from_array(ln2_scale,  'ln2_scale'),
        numpy_helper.from_array(ln2_bias,   'ln2_bias'),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8
model_bytes = model.SerializeToString()

INPUT = np.random.randn(1, D).astype(np.float32) * 0.01  # small values -> close LN outputs

# ── ORT opt vs no-opt ─────────────────────────────────────────────────────────
sess_opt = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
got = sess_opt.run(None, {'X': INPUT})[0]

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model_bytes, sess_options=opts,
                                 providers=['CPUExecutionProvider'])
expected = sess_ref.run(None, {'X': INPUT})[0]

max_diff = float(np.max(np.abs(got - expected)))

# ── numpy reference (shows what sequential computation gives) ─────────────────
def numpy_layernorm(x, scale, bias, axis=0, eps=1e-5):
    mean = np.mean(x, axis=axis, keepdims=True)
    var  = np.var(x,  axis=axis, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias

gathered  = embeddings[0:1]       # shape [1, D]
added     = gathered + INPUT       # [1, D]
ln1_ref   = numpy_layernorm(added, gln_scale, gln_bias, axis=0)
# TopK(k=1, axis=-1, largest=True): take max along last dim
tk_idx_ref = int(np.argmax(ln1_ref, axis=-1))
tk_val_ref = float(np.max(ln1_ref, axis=-1))
tiled_ref  = np.full((1, D), tk_val_ref, dtype=np.float32)
ref_out    = numpy_layernorm(tiled_ref, ln2_scale, ln2_bias, axis=0)

print("=== Bug v2-0006: Gather+LayerNorm fusion -> TopK index divergence ===")
print(f"Input shape: {INPUT.shape}")
print(f"\nORT_ENABLE_ALL output[:4]:  {got.ravel()[:4]}")
print(f"ORT_DISABLE_ALL output[:4]: {expected.ravel()[:4]}")
print(f"numpy ref output[:4]:       {ref_out.ravel()[:4]}")
print(f"\nmax_diff (OPT vs NOOPT):   {max_diff:.4e}")

# Show the LN1 output (the TopK input) to highlight near-ties
print(f"\nLayerNorm output (TopK input) max 4 values: {np.sort(ln1_ref.ravel())[-4:][::-1]}")
print(f"LayerNorm output (TopK input) argmax: {tk_idx_ref}")
print(f"Top-2 values differ by: {np.sort(ln1_ref.ravel())[-1] - np.sort(ln1_ref.ravel())[-2]:.4e}")
print(f"  (very close values create tie-breaking sensitivity after Gather+LN fusion)")

pass_flag = max_diff < 1e-4
print(f"\nPASS={pass_flag}")
if not pass_flag:
    print(f"BUG: Gather+LN fusion changes TopK result vs sequential path.")
else:
    print("ORT_ENABLE_ALL == ORT_DISABLE_ALL on this input.")
    print("Original full model (with larger embedding+input) showed rel-L2 ~5.4e2 vs pytorch.")
