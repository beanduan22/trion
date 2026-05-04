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
x   = np.random.randn(2, 512).astype(np.float32)
W   = np.random.randn(512, 64).astype(np.float32)
sh1 = np.array([1, 128], dtype=np.int64)
sh2 = np.array([2, 64],  dtype=np.int64)

nodes = [
    oh.make_node("MatMul",  ["X","W"],    ["mm"]),
    oh.make_node("Reshape", ["mm","sh1"], ["r1"]),
    oh.make_node("Reshape", ["r1","sh2"], ["Y"]),
]
graph = oh.make_graph(nodes, "redund_reshape",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[onh.from_array(W,"W"), onh.from_array(sh1,"sh1"), onh.from_array(sh2,"sh2")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print(f"ORT: {ref.ravel()[:4]}  (correct)")

try:
    net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    eager_out = net(torch.from_numpy(x)).detach().numpy()
    print(f"eager: {eager_out.ravel()[:4]}  (correct)")
    compiled = torch.compile(net)
    with torch.no_grad():
        out = compiled(torch.from_numpy(x))
    print("NOT reproduced: torch.compile succeeded"); sys.exit(1)
except Exception as e:
    print(f"BUG REPRODUCED: torch.compile crashes on redundant_reshape: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(0)
