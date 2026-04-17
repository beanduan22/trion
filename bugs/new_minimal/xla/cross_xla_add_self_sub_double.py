#!/usr/bin/env python3
"""
Bug: XLA diverges on add_self_sub_double
Compiler: XLA (via JAX)
Oracle:   ORT_DISABLE_ALL
Patterns: MatMul(x,W) -> Add(R) -> Add(R) -> Sub(R)
Root cause: XLA fp32 GEMM accumulation / optimization differs from ORT.
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
except ImportError:
    print("missing dep: jax"); sys.exit(2)

np.random.seed(42)
x = np.random.randn(2, 512).astype(np.float32)
W = np.random.randn(512, 64).astype(np.float32)
R = np.random.randn(2, 64).astype(np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W"],    ["mm"]),
    oh.make_node("Add",    ["mm","R"],   ["a1"]),
    oh.make_node("Add",    ["a1","R"],   ["a2"]),
    oh.make_node("Sub",    ["a2","R"],   ["Y"]),
]
graph = oh.make_graph(nodes, "test",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[onh.from_array(W,"W"), onh.from_array(R,"R")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}")

try:
    W_j = jnp.array(W); R_j = jnp.array(R)
    @jax.jit
    def fn(x_in):
        mm = jnp.matmul(x_in, W_j)
        a1 = mm + R_j
        a2 = a1 + R_j
        return a2 - R_j
    xla_out = np.array(fn(jnp.array(x)))
    max_abs = float(np.abs(ref.ravel() - xla_out.ravel()).max())
    print(f"XLA: {xla_out.ravel()[:4]}")
    print(f"max_abs={max_abs:.4f}")
    if max_abs > 0.01:
        print(f"BUG REPRODUCED: XLA add_self_sub_double (max_abs={max_abs:.4f})")
        sys.exit(0)
    print("NOT reproduced"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED (XLA error): {type(e).__name__}: {str(e)[:150]}")
    sys.exit(0)
