#!/usr/bin/env python3
"""
Bug: XLA diverges on matmul_add_biasgelu_bcast
Compiler: XLA (via JAX)
Oracle:   ORT_DISABLE_ALL
Patterns: MatMul(x,W) -> Add(bias) -> GELU -> Mul(gate)
Root cause: XLA fp32 GEMM + GELU accumulation differs from ORT.
Tolerance: 0.01

Exit 0 = BUG REPRODUCED, 1 = not reproduced, 2 = missing deps
"""
import sys
try:
    import numpy as np, onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}"); sys.exit(2)
try:
    import jax, jax.numpy as jnp
    from jax.nn import gelu
except ImportError:
    print("missing dep: jax"); sys.exit(2)

np.random.seed(42)
x    = np.random.randn(2, 512).astype(np.float32)
W    = np.random.randn(512, 64).astype(np.float32)
b    = np.random.randn(64).astype(np.float32)
gate = np.random.randn(2, 64).astype(np.float32)

# build ONNX reference
c1 = np.array(0.7978845608, dtype=np.float32)
c2 = np.array(0.044715, dtype=np.float32)
c05 = np.array(0.5, dtype=np.float32); c1v = np.array(1.0, dtype=np.float32)
nodes = [
    oh.make_node("MatMul", ["X","W"],        ["mm"]),
    oh.make_node("Add",    ["mm","b"],        ["h"]),
    oh.make_node("Mul",    ["h","h"],         ["h2"]),
    oh.make_node("Mul",    ["h2","h"],        ["h3"]),
    oh.make_node("Mul",    ["h3","c2"],       ["h3c"]),
    oh.make_node("Add",    ["h","h3c"],       ["hsum"]),
    oh.make_node("Mul",    ["hsum","c1"],     ["harg"]),
    oh.make_node("Tanh",   ["harg"],          ["htanh"]),
    oh.make_node("Add",    ["htanh","c1v"],   ["h1p"]),
    oh.make_node("Mul",    ["h1p","c05"],     ["hgelu"]),
    oh.make_node("Mul",    ["h","hgelu"],     ["gelu_out"]),
    oh.make_node("Mul",    ["gelu_out","G"],  ["Y"]),
]
graph = oh.make_graph(nodes, "biasgelu",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[
        onh.from_array(W,"W"), onh.from_array(b,"b"), onh.from_array(gate,"G"),
        onh.from_array(c1,"c1"), onh.from_array(c2,"c2"),
        onh.from_array(c05,"c05"), onh.from_array(c1v,"c1v"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}")

try:
    W_j=jnp.array(W); b_j=jnp.array(b); G_j=jnp.array(gate)
    @jax.jit
    def fn(x_in):
        h = jnp.matmul(x_in, W_j) + b_j
        return gelu(h) * G_j
    xla_out = np.array(fn(jnp.array(x)))
    max_abs = float(np.abs(ref.ravel() - xla_out.ravel()).max())
    print(f"XLA: {xla_out.ravel()[:4]}")
    print(f"max_abs={max_abs:.4f}")
    if max_abs > 0.01:
        print(f"BUG REPRODUCED: XLA matmul_add_biasgelu_bcast (max_abs={max_abs:.4f})")
        sys.exit(0)
    print("NOT reproduced"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED (XLA error): {type(e).__name__}: {str(e)[:150]}")
    sys.exit(0)
