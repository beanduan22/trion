#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_conv_k7_stride2
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : Conv2d(kernel=7, stride=2, padding=3) — ResNet-style stem —
             followed by Reshape(runtime_shape).
Root cause : The Conv2d runs fine through onnx2torch and ORT.  The trailing
             ONNX Reshape with a runtime-tensor shape converts (in onnx2torch)
             to torch.reshape(x, shape_tensor.tolist()).  Under torch.compile,
             shape_tensor becomes a FakeTensor and torch.Size() rejects it:
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
B, C_in, H, W = 1, 3, 64, 64
C_out = 8
x = np.random.randn(B, C_in, H, W).astype(np.float32)
w = (np.random.randn(C_out, C_in, 7, 7) * 0.05).astype(np.float32)

# After Conv(k=7, s=2, p=3): spatial is (64-7+2*3)/2 + 1 = 32
s = np.array([B, C_out, -1], dtype=np.int64)  # runtime shape → compile crash

nodes = [
    oh.make_node(
        "Conv",
        ["X", "W"],
        ["c"],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
    ),
    oh.make_node("Reshape", ["c", "s"], ["Y"]),
]
graph = oh.make_graph(
    nodes,
    "conv_k7_stride2",
    [oh.make_tensor_value_info("X", TP.FLOAT, [B, C_in, H, W])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [B, C_out, 32 * 32])],
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
