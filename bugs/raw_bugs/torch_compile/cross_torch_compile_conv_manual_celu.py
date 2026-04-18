#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_conv_manual_celu
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : Conv2d → manual CELU activation expressed as primitive ops:
               y = x                    where x >= 0
               y = alpha*(exp(x/alpha)-1) where x <  0
             The CELU decomposition uses Exp/Div/Sub/Mul/Greater/Where.
             Graph ends with Reshape(runtime_shape).
Root cause : The Conv + manual-CELU chain runs correctly in onnx2torch and
             ORT.  The trailing ONNX Reshape with a runtime-tensor shape
             converts (via onnx2torch) to torch.reshape(x, shape.tolist()).
             Under torch.compile, shape becomes a FakeTensor and torch.Size()
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
B, C_in, H, W = 1, 4, 16, 16
C_out = 4
ALPHA = 1.0
x = np.random.randn(B, C_in, H, W).astype(np.float32)
w = (np.random.randn(C_out, C_in, 3, 3) * 0.1).astype(np.float32)

alpha_t = np.array(ALPHA, dtype=np.float32)
zero_t = np.array(0.0, dtype=np.float32)
s = np.array([B, C_out, -1], dtype=np.int64)  # runtime shape triggers compile crash

nodes = [
    oh.make_node(
        "Conv", ["X", "W"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    ),
    # manual CELU: where(c >= 0, c, alpha * (exp(c/alpha) - 1))
    oh.make_node("Div", ["c", "alpha"], ["c_div"]),
    oh.make_node("Exp", ["c_div"], ["c_exp"]),
    oh.make_node("Sub", ["c_exp", "one"], ["c_exp_m1"]),
    oh.make_node("Mul", ["c_exp_m1", "alpha"], ["neg_branch"]),
    oh.make_node("Greater", ["c", "zero"], ["ge_mask"]),
    oh.make_node("Where", ["ge_mask", "c", "neg_branch"], ["celu"]),
    oh.make_node("Reshape", ["celu", "s"], ["Y"]),
]
inits = [
    onh.from_array(w, "W"),
    onh.from_array(alpha_t, "alpha"),
    onh.from_array(np.array(1.0, dtype=np.float32), "one"),
    onh.from_array(zero_t, "zero"),
    onh.from_array(s, "s"),
]
graph = oh.make_graph(
    nodes,
    "conv_manual_celu",
    [oh.make_tensor_value_info("X", TP.FLOAT, [B, C_in, H, W])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [B, C_out, H * W])],
    initializer=inits,
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
