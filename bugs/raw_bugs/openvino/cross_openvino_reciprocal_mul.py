#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_reciprocal_mul
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Reciprocal(x) -> Mul(W)  where x contains small values -> large reciprocals
Root cause : OV CPU handles Reciprocal followed by large-value MatMul with different
             fp32 precision than ORT. max_abs already reproduced at ~13767.
Tolerance  : 1.0

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys
try:
    import numpy as np, onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort, openvino as ov
except ImportError as e:
    print(f"missing dep: {e}"); sys.exit(2)

np.random.seed(42)
# small denominators produce large reciprocals
x = np.random.uniform(0.001, 0.1, (2, 16)).astype(np.float32)
W = np.random.randn(16, 8).astype(np.float32) * 10.0

nodes = [
    oh.make_node("Reciprocal", ["X"],    ["recip"]),
    oh.make_node("MatMul",     ["recip","W"], ["Y"]),
]
graph = oh.make_graph(nodes, "recip_mul",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,8])],
    initializer=[onh.from_array(W,"W")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

core = ov.Core()
comp = core.compile_model(core.read_model(mb, b""), "CPU")
ov_out = np.array(comp({"X": x})[comp.output(0)])

max_abs = float(np.abs(ref.ravel() - ov_out.ravel()).max())
print(f"ORT:      {ref.ravel()[:4]}")
print(f"OpenVINO: {ov_out.ravel()[:4]}")
print(f"max_abs={max_abs:.4f}")

if max_abs > 1.0:
    print(f"BUG REPRODUCED: OpenVINO reciprocal_mul (max_abs={max_abs:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
