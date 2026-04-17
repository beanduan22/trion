#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_matmul_add_layernorm
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : MatMul -> Add(bias) -> LayerNormalization(axis=-1) ->
             Reshape(runtime_shape).  Classic transformer block head.
Root cause : The MatMul+Bias+LayerNorm chain lowers cleanly through
             onnx2torch and ORT.  The trailing Reshape uses a runtime-tensor
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
B, K, D = 4, 64, 32
x = np.random.randn(B, K).astype(np.float32)
w = (np.random.randn(K, D) * 0.1).astype(np.float32)
bias = np.random.randn(D).astype(np.float32) * 0.1
gamma = np.ones(D, dtype=np.float32)
beta = np.zeros(D, dtype=np.float32)
s = np.array([B, D], dtype=np.int64)

nodes = [
    oh.make_node("MatMul", ["X", "W"], ["mm"]),
    oh.make_node("Add", ["mm", "bias"], ["preln"]),
    oh.make_node("LayerNormalization", ["preln", "gamma", "beta"], ["ln"], axis=-1, epsilon=1e-5),
    oh.make_node("Reshape", ["ln", "s"], ["Y"]),
]
graph = oh.make_graph(
    nodes,
    "matmul_add_layernorm",
    [oh.make_tensor_value_info("X", TP.FLOAT, [B, K])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [B, D])],
    initializer=[
        onh.from_array(w, "W"),
        onh.from_array(bias, "bias"),
        onh.from_array(gamma, "gamma"),
        onh.from_array(beta, "beta"),
        onh.from_array(s, "s"),
    ],
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(
    None, {"X": x}
)[0]
print(f"ORT: shape={ref.shape}  first4={ref.ravel()[:4]}")

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
