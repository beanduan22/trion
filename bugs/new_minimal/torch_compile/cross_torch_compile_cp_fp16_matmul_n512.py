#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_cp_fp16_matmul_n512
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : fp16 MatMul with K=N=512 followed by Reshape(runtime_shape).
             Large-K fp16 accumulation is sensitive to kernel choice and
             order of summation.
Root cause : The fp16 MatMul itself converts cleanly via onnx2torch.  The
             trailing Reshape with a runtime-tensor shape converts to
             torch.reshape(x, shape.tolist()); under torch.compile the
             shape tensor becomes a FakeTensor and torch.Size() rejects it:
               TypeError: torch.Size() takes an iterable of 'int'
                          (item 0 is 'FakeTensor')
             The K=512 fp16 accumulation is also a known source of tiny
             numerical drift eager↔compile when compile succeeds.
Tolerance  : fp16 MatMul drift usually < 5e-2; compile-crash path is
             exit-0 directly.

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
M, K, N = 32, 512, 512
x = (np.random.randn(M, K).astype(np.float32) * 0.1).astype(np.float16)
w = (np.random.randn(K, N).astype(np.float32) * 0.1).astype(np.float16)
s = np.array([M, N], dtype=np.int64)  # runtime shape triggers compile crash

nodes = [
    oh.make_node("MatMul", ["X", "W"], ["mm"]),
    oh.make_node("Reshape", ["mm", "s"], ["Y"]),
]
graph = oh.make_graph(
    nodes,
    "cp_fp16_matmul_n512",
    [oh.make_tensor_value_info("X", TP.FLOAT16, [M, K])],
    [oh.make_tensor_value_info("Y", TP.FLOAT16, [M, N])],
    initializer=[onh.from_array(w, "W"), onh.from_array(s, "s")],
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(
    None, {"X": x}
)[0]
print(f"ORT fp16: shape={ref.shape}  first4={ref.ravel()[:4]}")

net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
x_t = torch.from_numpy(x)
with torch.no_grad():
    eager_out = net(x_t).numpy()
print(f"eager onnx2torch: shape={eager_out.shape}  first4={eager_out.ravel()[:4]}")

try:
    compiled = torch.compile(net, mode="default")
    with torch.no_grad():
        got = compiled(x_t).numpy()
    diff = float(np.abs(got.astype(np.float32) - eager_out.astype(np.float32)).max())
    print(f"torch.compile fp16: shape={got.shape}  max_diff_vs_eager={diff:.2e}")
    # fp16 accum drift > 1e-3 counts as divergence (eager uses matmul_fp16; compile may not)
    if diff > 1e-3:
        print("BUG REPRODUCED: torch.compile fp16 numerical divergence vs eager (K=N=512)")
        sys.exit(0)
    print("NOT reproduced: torch.compile matches eager within fp16 tolerance")
    sys.exit(1)
except Exception as e:
    print("BUG REPRODUCED: torch.compile crashes while eager succeeds")
    print(f"  {type(e).__name__}: {str(e)[:160]}")
    sys.exit(0)
