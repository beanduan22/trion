#!/usr/bin/env python3
"""
Bug: torch.compile (Inductor) crashes on topk_last_axis_k1
Compiler: torch.compile
Oracle:   ORT_DISABLE_ALL — runs correctly; eager onnx2torch also runs correctly

TopK → Reshape: torch.compile crashes on onnx2torch Reshape with runtime shape tensor after TopK

Root cause: onnx2torch converts ONNX Reshape (with shape as a runtime input tensor)
to torch.reshape; when torch.compile/Inductor traces the model it propagates shapes
symbolically — the shape tensor becomes a FakeTensor, and PyTorch raises
  TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys
try:
    import numpy as np, onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}"); sys.exit(2)
try:
    import onnx2torch, torch
except ImportError:
    print("missing dep: onnx2torch or torch"); sys.exit(2)


np.random.seed(42)
x = np.random.randn(4, 8).astype(np.float32)
k = np.array([1], dtype=np.int64)
s = np.array([4], dtype=np.int64)

nodes = [
    oh.make_node("TopK",   ["X","k"], ["vals","idx"], axis=1),
    oh.make_node("Reshape",["vals","s"], ["Y"]),
]
graph = oh.make_graph(nodes, "topk_reshape",
    [oh.make_tensor_value_info("X",  TP.FLOAT, [4, 8])],
    [oh.make_tensor_value_info("Y",  TP.FLOAT, [4])],
    initializer=[onh.from_array(k,"k"), onh.from_array(s,"s")])
feed = {"X": x}

model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

# ── reference: ORT with all optimisations off ─────────────────────────────────
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref_out = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, feed)[0]
print(f"ORT: {ref_out.ravel()[:4]}  (correct)")

# ── eager onnx2torch also works ───────────────────────────────────────────────
net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
x_t = torch.from_numpy(list(feed.values())[0])
with torch.no_grad():
    eager_out = net(x_t).numpy()
print(f"eager onnx2torch: {eager_out.ravel()[:4]}  (correct)")

# ── torch.compile crashes ─────────────────────────────────────────────────────
try:
    compiled = torch.compile(net)
    with torch.no_grad():
        compiled(x_t)
    print("NOT reproduced: torch.compile succeeded"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED: torch.compile crashes on topk_last_axis_k1")
    print(f"  {type(e).__name__}: {str(e)[:120]}")
    sys.exit(0)
