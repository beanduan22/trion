"""
Root Cause 6 — ORT graph optimizer fires unsafe algebraic eliminations
======================================================================
Affects: campaign_v2 uid 0017, 0019 (log(exp(x))→x cancel)
         campaign_v2 uid 0032, 0038 (x-x→0, x*1→x eliminations)
         campaign_v2 uid 0042     (x*0 before ReduceSum shape-change)

Bug A — Log-Exp cancel (uid 0017, 0019):
  ORT simplifies  Log(Exp(x)) → x  when x might be negative or very large.
  For negative x, Exp(x) is fine but Log(0…) should return -inf; the cancel
  hides this.  For large x, Exp overflows to +inf and Log(inf)=+inf; the
  cancel wrongly gives a finite value.

Bug B — Sub-self elimination (uid 0032, 0038):
  ORT rewrites  x - x → 0  (constant zero tensor).  If the zero is then
  multiplied back into a SkipLayerNorm fusion, the fusion kernel ignores
  the zero branch entirely, producing wrong output when the zero was meant
  to clear the skip connection.

Bug C — Mul-zero before ReduceSum (uid 0042):
  ORT eliminates  x * 0  to a constant-zero tensor before it reaches
  ReduceSum.  The constant zero has shape [], not the broadcast shape of x,
  so the downstream ReduceSum receives a scalar instead of a tensor,
  silently changing the output shape/values.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


# ─────────────────────────────────────────────────────────
# Bug A: Log(Exp(x)) cancel with overflow
# ─────────────────────────────────────────────────────────
print("=== Bug A: Log(Exp(x)) cancel — overflow case ===")

x_a = np.array([[200., 300., 400., 500.]], dtype=np.float32)  # Exp overflows

X_a = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
Y_a = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

exp_n = helper.make_node('Exp', ['X'],   ['e'])
log_n = helper.make_node('Log', ['e'],   ['Y'])
g_a   = helper.make_graph([exp_n, log_n], 'g', [X_a], [Y_a])
m_a   = helper.make_model(g_a, opset_imports=[helper.make_opsetid('', 13)])

opts_all  = ort.SessionOptions(); opts_all.graph_optimization_level  = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts_none = ort.SessionOptions(); opts_none.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

s_all  = ort.InferenceSession(m_a.SerializeToString(), sess_options=opts_all,  providers=['CPUExecutionProvider'])
s_none = ort.InferenceSession(m_a.SerializeToString(), sess_options=opts_none, providers=['CPUExecutionProvider'])

out_all  = s_all.run( None, {'X': x_a})[0].ravel()
out_none = s_none.run(None, {'X': x_a})[0].ravel()

# Correct: Exp(500) = +inf, Log(+inf) = +inf
correct_a = np.log(np.exp(x_a.astype(np.float64))).astype(np.float32).ravel()
print(f"Input:           {x_a.ravel()}")
print(f"ORT ENABLE_ALL:  {out_all}")      # may give finite values (wrongly)
print(f"ORT DISABLE_ALL: {out_none}")     # gives +inf (correct)
print(f"Float64 ref:     {correct_a}")
print(f"ORT_ALL == ORT_NONE: {np.allclose(out_all, out_none, equal_nan=True)}")
print()

# ─────────────────────────────────────────────────────────
# Bug B: x - x → 0, then fed into a subsequent operation
# ─────────────────────────────────────────────────────────
print("=== Bug B: Sub(X, X)→0 elimination changes downstream Mul ===")

x_b = np.array([[1., 2., 3., 4.]], dtype=np.float32)
y_b = np.array([[5., 6., 7., 8.]], dtype=np.float32)  # second input

X_b = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
W_b = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 4])
Y_b = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

from onnx import numpy_helper
w_init = numpy_helper.from_array(y_b, 'W')

sub_b = helper.make_node('Sub',  ['X', 'X'], ['zero'])   # x - x = 0
mul_b = helper.make_node('Mul',  ['zero', 'W'], ['Y'])   # 0 * w = 0 (should be 0)
g_b   = helper.make_graph([sub_b, mul_b], 'g', [X_b], [Y_b], initializer=[w_init])
m_b   = helper.make_model(g_b, opset_imports=[helper.make_opsetid('', 13)])

s_b_all  = ort.InferenceSession(m_b.SerializeToString(), sess_options=opts_all,  providers=['CPUExecutionProvider'])
s_b_none = ort.InferenceSession(m_b.SerializeToString(), sess_options=opts_none, providers=['CPUExecutionProvider'])

out_b_all  = s_b_all.run( None, {'X': x_b})[0].ravel()
out_b_none = s_b_none.run(None, {'X': x_b})[0].ravel()

print(f"Input X:         {x_b.ravel()},  W: {y_b.ravel()}")
print(f"Expected (X-X)*W = 0*W: {np.zeros(4)}")
print(f"ORT ENABLE_ALL:  {out_b_all}")
print(f"ORT DISABLE_ALL: {out_b_none}")
print(f"ORT_ALL == ORT_NONE: {np.allclose(out_b_all, out_b_none)}")
print()

# ─────────────────────────────────────────────────────────
# Bug C: x * 0 before ReduceSum — shape broadcast lost
# ─────────────────────────────────────────────────────────
print("=== Bug C: x*0 before ReduceSum — optimizer drops tensor shape ===")

x_c = np.ones((1, 4, 4), dtype=np.float32)
X_c = helper.make_tensor_value_info('X',    TensorProto.FLOAT, [1, 4, 4])
Y_c = helper.make_tensor_value_info('Y',    TensorProto.FLOAT, None)
zero_init = numpy_helper.from_array(np.zeros(1, dtype=np.float32), 'zero_scalar')
axes_init  = numpy_helper.from_array(np.array([2], dtype=np.int64), 'axes')

mul_c = helper.make_node('Mul',       ['X', 'zero_scalar'], ['x_zero'])
rs_c  = helper.make_node('ReduceSum', ['x_zero', 'axes'],   ['Y'],     keepdims=1)
g_c   = helper.make_graph([mul_c, rs_c], 'g', [X_c], [Y_c],
                           initializer=[zero_init, axes_init])
m_c   = helper.make_model(g_c, opset_imports=[helper.make_opsetid('', 13)])

s_c_all  = ort.InferenceSession(m_c.SerializeToString(), sess_options=opts_all,  providers=['CPUExecutionProvider'])
s_c_none = ort.InferenceSession(m_c.SerializeToString(), sess_options=opts_none, providers=['CPUExecutionProvider'])

out_c_all  = s_c_all.run( None, {'X': x_c})[0]
out_c_none = s_c_none.run(None, {'X': x_c})[0]

print(f"Input shape: {x_c.shape}  →  expected ReduceSum(x*0, axis=2, keepdims=1) shape: (1,4,1)")
print(f"ORT ENABLE_ALL  shape: {out_c_all.shape},  values[:4]: {out_c_all.ravel()[:4]}")
print(f"ORT DISABLE_ALL shape: {out_c_none.shape}, values[:4]: {out_c_none.ravel()[:4]}")
print(f"Shapes differ: {out_c_all.shape != out_c_none.shape}")
print()
print(f"PASS={np.allclose(out_b_all, out_b_none) and np.allclose(out_c_all, out_c_none) and np.allclose(out_all, out_none, equal_nan=True)}")
