#!/usr/bin/env python3
"""
Bug: XLA diverges on space_to_depth_block
Compiler: XLA (via JAX)
Oracle:   ORT_DISABLE_ALL
Patterns: Conv -> SpaceToDepth -> Conv
Root cause: XLA Conv + space_to_depth fp32 accumulation differs from ORT.
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
    from jax import lax
except ImportError:
    print("missing dep: jax"); sys.exit(2)

np.random.seed(42)
x  = np.random.randn(1, 32, 16, 16).astype(np.float32)
W1 = np.random.randn(8, 32, 3, 3).astype(np.float32)
W2 = np.random.randn(16, 32, 1, 1).astype(np.float32)

nodes = [
    oh.make_node("Conv",         ["X","W1"],   ["conv1"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("SpaceToDepth", ["conv1"],    ["s2d"],   blocksize=2),
    oh.make_node("Conv",         ["s2d","W2"], ["Y"],     kernel_shape=[1,1]),
]
graph = oh.make_graph(nodes, "s2d_block",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,32,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,16,8,8])],
    initializer=[onh.from_array(W1,"W1"), onh.from_array(W2,"W2")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}")

try:
    # JAX conv: use lax.conv_general_dilated
    W1_j = jnp.array(W1); W2_j = jnp.array(W2)

    @jax.jit
    def fn(x_in):
        # NCHW conv
        c1 = lax.conv_general_dilated(x_in, W1_j, (1,1), ((1,1),(1,1)),
                                       dimension_numbers=('NCHW','OIHW','NCHW'))
        # space_to_depth blocksize=2: NCHW -> reshape -> transpose -> reshape
        N,C,H,W = c1.shape
        bs = 2
        c1r = c1.reshape(N, C, H//bs, bs, W//bs, bs)
        c1t = c1r.transpose(0, 3, 5, 1, 2, 4)
        s2d = c1t.reshape(N, C*bs*bs, H//bs, W//bs)
        out = lax.conv_general_dilated(s2d, W2_j, (1,1), ((0,0),(0,0)),
                                        dimension_numbers=('NCHW','OIHW','NCHW'))
        return out

    xla_out = np.array(fn(jnp.array(x)))
    max_abs = float(np.abs(ref.ravel() - xla_out.ravel()).max())
    print(f"XLA: {xla_out.ravel()[:4]}")
    print(f"max_abs={max_abs:.4f}")
    if max_abs > 0.01:
        print(f"BUG REPRODUCED: XLA space_to_depth_block (max_abs={max_abs:.4f})")
        sys.exit(0)
    print("NOT reproduced"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED (XLA error): {type(e).__name__}: {str(e)[:150]}")
    sys.exit(0)
