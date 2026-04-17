#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_reduce_l2_last
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : L2-normalize last axis decomposed as ReduceL2(axes=[-1],
             keepdims=1) -> Add(epsilon) -> Div(x, norm).  Then
             Reshape(runtime_shape).
Root cause : The ReduceL2 + Add + Div chain converts cleanly eager/ORT.
             The trailing Reshape uses a runtime-tensor shape; under
             torch.compile that tensor becomes a FakeTensor and
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
B, N, D = 2, 8, 16
x = np.random.randn(B, N, D).astype(np.float32)
eps = np.array(1e-12, dtype=np.float32)
s = np.array([B, N, D], dtype=np.int64)  # runtime shape → compile crash

# Use opset 13 ReduceL2 (axes as attribute).  Equivalent to L2 norm along last dim.
nodes = [
    oh.make_node("ReduceL2", ["X"], ["norm"], axes=[-1], keepdims=1),
    oh.make_node("Add", ["norm", "eps"], ["norm_safe"]),
    oh.make_node("Div", ["X", "norm_safe"], ["l2n"]),
    oh.make_node("Reshape", ["l2n", "s"], ["Y"]),
]
graph = oh.make_graph(
    nodes,
    "reduce_l2_last",
    [oh.make_tensor_value_info("X", TP.FLOAT, [B, N, D])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [B, N, D])],
    initializer=[
        onh.from_array(eps, "eps"),
        onh.from_array(s, "s"),
    ],
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(
    None, {"X": x}
)[0]
norms = np.linalg.norm(ref, axis=-1)
print(
    f"ORT: shape={ref.shape}  first4={ref.ravel()[:4]}  "
    f"row_norm_min={norms.min():.3f} max={norms.max():.3f} (expect ~1)"
)

net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
x_t = torch.from_numpy(x)
with torch.no_grad():
    eager_out = net(x_t).numpy()
print(f"eager onnx2torch: shape={eager_out.shape}  first4={eager_out.ravel()[:4]}")

try:
    compiled = torch.compile(net, mode="default")
    with torch.no_grad():
        got = compiled(x_t).numpy()
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
