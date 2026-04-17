#!/usr/bin/env python3
"""
Bug ID     : cross_torch_compile_ia_sat_sub_uint8_underflow
Compiler   : torch.compile (Inductor backend), torch 2.9.x
Oracle     : ORT_DISABLE_ALL and eager onnx2torch — both run correctly
Patterns   : Simulated saturating uint8 subtract via Cast(uint8->int32),
             Sub, Clip(0,255), Cast(int32->uint8) — guards against the
             natural uint8 wrap-around (e.g. 3 - 5 = 254 in raw uint8
             arithmetic, 0 when saturated).  Then Reshape(runtime_shape).
Root cause : The Cast/Sub/Clip/Cast chain is straightforward eager/ORT.
             The trailing Reshape uses a runtime-tensor shape; under
             torch.compile that tensor becomes a FakeTensor and
             torch.Size() rejects it:
               TypeError: torch.Size() takes an iterable of 'int'
                          (item 0 is 'FakeTensor')
Tolerance  : exact integer equality on compare paths.

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
N = 32
# craft values that would underflow uint8: 3 - 5, 0 - 200, 10 - 250
a_u8 = np.array([3, 0, 10, 200] * (N // 4), dtype=np.uint8)
b_u8 = np.array([5, 200, 250, 150] * (N // 4), dtype=np.uint8)
s = np.array([N], dtype=np.int64)

lo = np.array(0, dtype=np.int32)
hi = np.array(255, dtype=np.int32)

nodes = [
    oh.make_node("Cast", ["A"], ["a32"], to=TP.INT32),
    oh.make_node("Cast", ["B"], ["b32"], to=TP.INT32),
    oh.make_node("Sub", ["a32", "b32"], ["diff"]),
    oh.make_node("Clip", ["diff", "lo", "hi"], ["sat"]),
    oh.make_node("Cast", ["sat"], ["out_u8"], to=TP.UINT8),
    oh.make_node("Reshape", ["out_u8", "s"], ["Y"]),
]
graph = oh.make_graph(
    nodes,
    "ia_sat_sub_uint8_underflow",
    [
        oh.make_tensor_value_info("A", TP.UINT8, [N]),
        oh.make_tensor_value_info("B", TP.UINT8, [N]),
    ],
    [oh.make_tensor_value_info("Y", TP.UINT8, [N])],
    initializer=[
        onh.from_array(lo, "lo"),
        onh.from_array(hi, "hi"),
        onh.from_array(s, "s"),
    ],
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(
    None, {"A": a_u8, "B": b_u8}
)[0]
print(f"ORT sat_sub: first4={ref[:4].tolist()} (expect 0s via saturation)")

net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
a_t, b_t = torch.from_numpy(a_u8), torch.from_numpy(b_u8)
with torch.no_grad():
    eager_out = net(a_t, b_t).numpy()
print(f"eager onnx2torch: first4={eager_out[:4].tolist()}")

try:
    compiled = torch.compile(net, mode="default")
    with torch.no_grad():
        got = compiled(a_t, b_t).numpy()
    diff = int(np.abs(got.astype(np.int32) - eager_out.astype(np.int32)).max())
    print(f"torch.compile: first4={got[:4].tolist()}  max_diff={diff}")
    if diff > 0:
        print("BUG REPRODUCED: torch.compile numerical divergence vs eager")
        sys.exit(0)
    print("NOT reproduced: torch.compile matches eager")
    sys.exit(1)
except Exception as e:
    print("BUG REPRODUCED: torch.compile crashes while eager succeeds")
    print(f"  {type(e).__name__}: {str(e)[:160]}")
    sys.exit(0)
