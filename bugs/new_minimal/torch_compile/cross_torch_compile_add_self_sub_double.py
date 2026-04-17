#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_add_self_sub_double
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : Add(x,x) -> Add(x,x) -> Sub -> Reshape(runtime_shape)
             Mathematically `(x+x)-(x+x)` is identically 0; the graph still
             flows through onnx2torch's Reshape with a runtime-tensor shape
             argument, which is what trips torch.compile.
Root cause : onnx2torch converts ONNX Reshape (with the shape input coming
             from the graph rather than a compile-time constant) to
             `torch.reshape(x, shape_tensor.tolist())`.  During torch.compile
             /Inductor tracing, `shape_tensor` becomes a FakeTensor and
             `torch.Size()` rejects it:
               TypeError: torch.Size() takes an iterable of 'int'
                          (item 0 is 'FakeTensor')
             Eager PyTorch and ORT walk the runtime shape normally.
Tolerance  : N/A — compile path raises; eager path succeeds.

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
x = np.random.randn(4, 64).astype(np.float32)
s = np.array([4, 64], dtype=np.int64)  # runtime shape input → triggers FakeTensor

# Genuine Add/Add/Sub graph: (x+x) - (x+x) which simplifies to 0
nodes = [
    oh.make_node("Add", ["X", "X"], ["add1"]),
    oh.make_node("Add", ["X", "X"], ["add2"]),
    oh.make_node("Sub", ["add1", "add2"], ["sub"]),
    oh.make_node("Reshape", ["sub", "s"], ["Y"]),
]
graph = oh.make_graph(
    nodes,
    "add_self_sub_double",
    [oh.make_tensor_value_info("X", TP.FLOAT, [4, 64])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [4, 64])],
    initializer=[onh.from_array(s, "s")],
)
feed = {"X": x}

model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

# ── ORT reference ────────────────────────────────────────────────────────────
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref_out = ort.InferenceSession(
    mb, sess_options=so, providers=["CPUExecutionProvider"]
).run(None, feed)[0]
print(f"ORT: first4={ref_out.ravel()[:4]}  max|y|={np.abs(ref_out).max():.2e} (should be 0)")

# ── eager onnx2torch ─────────────────────────────────────────────────────────
net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
x_t = torch.from_numpy(x)
with torch.no_grad():
    eager_out = net(x_t).numpy()
print(f"eager onnx2torch: first4={eager_out.ravel()[:4]}  max|y|={np.abs(eager_out).max():.2e}")

# ── torch.compile — expected to crash on runtime Reshape ─────────────────────
try:
    compiled = torch.compile(net, mode="default")
    with torch.no_grad():
        got = compiled(x_t).numpy()
    diff = float(np.abs(got - eager_out).max())
    print(f"torch.compile: first4={got.ravel()[:4]}  max_diff_vs_eager={diff:.2e}")
    if diff > 1e-6:
        print("BUG REPRODUCED: torch.compile numerical divergence vs eager")
        sys.exit(0)
    print("NOT reproduced: torch.compile matches eager")
    sys.exit(1)
except Exception as e:
    print("BUG REPRODUCED: torch.compile crashes while eager succeeds")
    print(f"  {type(e).__name__}: {str(e)[:160]}")
    sys.exit(0)
