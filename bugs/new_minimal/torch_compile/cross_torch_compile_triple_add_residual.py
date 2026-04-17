#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_triple_add_residual
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : Triple Add residual: out = a + b + c + d expressed as
             Add(Add(Add(a,b),c),d) — a left-folded residual chain — then
             Reshape(runtime_shape).
Root cause : The three-node Add chain lowers cleanly eager/ORT.  The
             trailing Reshape uses a runtime-tensor shape; under
             torch.compile that tensor becomes a FakeTensor and torch.Size()
             rejects it:
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
B, F = 4, 32
a = np.random.randn(B, F).astype(np.float32)
b = np.random.randn(B, F).astype(np.float32)
c = np.random.randn(B, F).astype(np.float32)
d = np.random.randn(B, F).astype(np.float32)
s = np.array([B, F], dtype=np.int64)

nodes = [
    oh.make_node("Add", ["A", "B"], ["ab"]),
    oh.make_node("Add", ["ab", "C"], ["abc"]),
    oh.make_node("Add", ["abc", "D"], ["abcd"]),
    oh.make_node("Reshape", ["abcd", "s"], ["Y"]),
]
graph = oh.make_graph(
    nodes,
    "triple_add_residual",
    [
        oh.make_tensor_value_info("A", TP.FLOAT, [B, F]),
        oh.make_tensor_value_info("B", TP.FLOAT, [B, F]),
        oh.make_tensor_value_info("C", TP.FLOAT, [B, F]),
        oh.make_tensor_value_info("D", TP.FLOAT, [B, F]),
    ],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [B, F])],
    initializer=[onh.from_array(s, "s")],
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

feed = {"A": a, "B": b, "C": c, "D": d}
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(
    None, feed
)[0]
print(f"ORT: shape={ref.shape}  first4={ref.ravel()[:4]}")

net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
a_t, b_t, c_t, d_t = (torch.from_numpy(x) for x in (a, b, c, d))
with torch.no_grad():
    eager_out = net(a_t, b_t, c_t, d_t).numpy()
print(f"eager onnx2torch: shape={eager_out.shape}  first4={eager_out.ravel()[:4]}")

try:
    compiled = torch.compile(net, mode="default")
    with torch.no_grad():
        got = compiled(a_t, b_t, c_t, d_t).numpy()
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
