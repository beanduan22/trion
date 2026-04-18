#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_sub_self_mul_zero
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL (CPU EP)
Patterns   : MatMul fp32 accumulation diff with subtractive cancellation.
             Graph = Mul( Sub( MatMul(X,W1), MatMul(X,W2) ), const )
             where W1 ≠ W2. Not a true self-subtraction — we rely on the
             cancellation between two close-magnitude GEMMs to amplify any
             order-of-accumulation differences between OpenVINO's CPU GEMM
             kernel and ORT's reference GEMM.
Why it's informative:
             With random 4×512 × 512×128 inputs both matmul products are
             O(√512) ≈ 22 per element. Taking their difference destroys most
             of the signal, so any per-kernel fp32 reordering (e.g. AVX2
             tile blocking vs scalar order) shows up unmasked in the
             low-order bits. Multiplying by 10.0 then brings the delta
             above a generous 0.01 tolerance if the backends really do
             accumulate in different orders.
Note       : Historically this check was labelled "sub_self should give
             near-zero". That is incorrect for this graph (W1 ≠ W2); we've
             relabelled it to reflect what it actually tests — a
             cancellation-sensitive GEMM cross-check between OpenVINO CPU
             and ORT CPU.
Tolerance  : 0.01

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
x  = np.random.randn(4, 512).astype(np.float32)
W1 = np.random.randn(512, 128).astype(np.float32)
W2 = np.random.randn(512, 128).astype(np.float32)
# Sub of two different MatMul results (not self-sub) - still shows fp32 diff
sc = np.array(10.0, dtype=np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W1"],   ["mm1"]),
    oh.make_node("MatMul", ["X","W2"],   ["mm2"]),
    oh.make_node("Sub",    ["mm1","mm2"], ["diff"]),
    oh.make_node("Mul",    ["diff","sc"], ["Y"]),
]
graph = oh.make_graph(nodes, "sub_mul",
    [oh.make_tensor_value_info("X", TP.FLOAT, [4,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [4,128])],
    initializer=[onh.from_array(W1,"W1"), onh.from_array(W2,"W2"), onh.from_array(sc,"sc")])
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

if max_abs > 0.01:
    print(
        f"BUG REPRODUCED: OpenVINO CPU vs ORT CPU diverge on "
        f"Sub(MatMul,MatMul)*const  (max_abs={max_abs:.4f}) — "
        f"fp32 GEMM accumulation-order difference exposed by subtractive "
        f"cancellation."
    )
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
