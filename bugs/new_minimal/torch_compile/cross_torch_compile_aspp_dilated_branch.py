#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_aspp_dilated_branch
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : ASPP head: parallel Conv2d branches with dilation=1,6,12,18
             followed by Add fusion, then Reshape(runtime_shape).
Root cause : The four dilated Conv2d branches compute & sum normally under
             eager onnx2torch and ORT.  The trailing ONNX Reshape with a
             runtime-tensor shape input converts (via onnx2torch) to
             torch.reshape(x, shape_tensor.tolist()).  Under torch.compile,
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
B, C_in, H, W = 1, 8, 16, 16
C_out = 4
x = np.random.randn(B, C_in, H, W).astype(np.float32)

# four 3x3 convs, dilation in {1, 6, 12, 18} — each with matching "same" padding
def w_init(name):
    return onh.from_array(
        (np.random.randn(C_out, C_in, 3, 3) * 0.05).astype(np.float32), name
    )

dilations = [1, 6, 12, 18]
nodes = []
branch_outs = []
for i, d in enumerate(dilations):
    pad = d  # SAME with k=3 dilated: pad = d on each side
    nodes.append(
        oh.make_node(
            "Conv",
            [f"X", f"Wb{i}"],
            [f"b{i}"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[pad, pad, pad, pad],
            dilations=[d, d],
        )
    )
    branch_outs.append(f"b{i}")

# fuse: b0 + b1 + b2 + b3
nodes.append(oh.make_node("Add", [branch_outs[0], branch_outs[1]], ["s01"]))
nodes.append(oh.make_node("Add", [branch_outs[2], branch_outs[3]], ["s23"]))
nodes.append(oh.make_node("Add", ["s01", "s23"], ["fused"]))

# Reshape with a runtime-tensor shape input — the compile trip
s = np.array([B, C_out, H * W], dtype=np.int64)
nodes.append(oh.make_node("Reshape", ["fused", "s"], ["Y"]))

inits = [w_init(f"Wb{i}") for i in range(4)] + [onh.from_array(s, "s")]
graph = oh.make_graph(
    nodes,
    "aspp_dilated_branch",
    [oh.make_tensor_value_info("X", TP.FLOAT, [B, C_in, H, W])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [B, C_out, H * W])],
    initializer=inits,
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref_out = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(
    None, {"X": x}
)[0]
print(f"ORT: shape={ref_out.shape}  first4={ref_out.ravel()[:4]}")

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
