#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_multi_query_attention
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : Multi-Query Attention — Q has H heads; K and V share a single
             head (H_KV=1) which is broadcast across Q's heads.  Then QK^T
             scaled softmax, attn @ V, finally Reshape(runtime_shape).
Root cause : The MQA graph (MatMul, Div, Softmax, Transpose, broadcast) is
             fine eager/ORT.  The trailing Reshape uses a runtime-tensor
             shape; under torch.compile that tensor becomes a FakeTensor and
             torch.Size() rejects it:
               TypeError: torch.Size() takes an iterable of 'int'
                          (item 0 is 'FakeTensor')
Tolerance  : N/A — compile raises; eager/ORT succeed.

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys

try:
    import numpy as np
    import onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)
try:
    import onnx2torch
    import torch
except ImportError:
    print("missing dep: onnx2torch or torch")
    sys.exit(2)


np.random.seed(42)
B, H, T, D = 1, 8, 4, 16
q = np.random.randn(B, H, T, D).astype(np.float32)
# MQA: K/V have single head; tensor shape (B, 1, T, D). Will broadcast.
k = np.random.randn(B, 1, T, D).astype(np.float32)
v = np.random.randn(B, 1, T, D).astype(np.float32)

# broadcast via Expand
expand_shape = np.array([B, H, T, D], dtype=np.int64)
scale = np.array(1.0 / np.sqrt(D), dtype=np.float32)
s = np.array([B, H, T * D], dtype=np.int64)  # runtime reshape → compile crash

nodes = [
    oh.make_node("Expand", ["K", "expand_shape"], ["k_b"]),
    oh.make_node("Expand", ["V", "expand_shape"], ["v_b"]),
    oh.make_node("Transpose", ["k_b"], ["kT"], perm=[0, 1, 3, 2]),
    oh.make_node("MatMul", ["Q", "kT"], ["scores"]),
    oh.make_node("Mul", ["scores", "scale"], ["scaled"]),
    oh.make_node("Softmax", ["scaled"], ["attn"], axis=-1),
    oh.make_node("MatMul", ["attn", "v_b"], ["out"]),
    oh.make_node("Reshape", ["out", "s_out"], ["Y"]),
]
inits = [
    onh.from_array(expand_shape, "expand_shape"),
    onh.from_array(scale, "scale"),
    onh.from_array(s, "s_out"),
]
graph = oh.make_graph(
    nodes,
    "mqa",
    [
        oh.make_tensor_value_info("Q", TP.FLOAT, [B, H, T, D]),
        oh.make_tensor_value_info("K", TP.FLOAT, [B, 1, T, D]),
        oh.make_tensor_value_info("V", TP.FLOAT, [B, 1, T, D]),
    ],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [B, H, T * D])],
    initializer=inits,
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

feed = {"Q": q, "K": k, "V": v}
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(
    None, feed
)[0]
print(f"ORT: shape={ref.shape}  first4={ref.ravel()[:4]}")

net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
q_t, k_t, v_t = torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v)
with torch.no_grad():
    eager_out = net(q_t, k_t, v_t).numpy()
print(f"eager onnx2torch: shape={eager_out.shape}  first4={eager_out.ravel()[:4]}")

try:
    compiled = torch.compile(net, mode="default")
    with torch.no_grad():
        got = compiled(q_t, k_t, v_t).numpy()
    diff = float(np.abs(got - eager_out).max())
    print(f"torch.compile: shape={got.shape}  max_diff_vs_eager={diff:.2e}")
    if diff > 1e-5:
        print("BUG REPRODUCED: torch.compile numerical divergence vs eager")
        sys.exit(0)
    print("NOT reproduced: torch.compile matches eager")
    sys.exit(1)
except Exception as e:
    print("BUG REPRODUCED: torch.compile crashes while eager succeeds")
    print(f"  {type(e).__name__}: {str(e)[:160]}")
    sys.exit(0)
