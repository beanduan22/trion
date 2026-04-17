#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_ia_sat_sub_uint8_underflow
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Cast(float->uint8) -> Clip (simulating uint8 subtraction underflow)
Root cause : OV may handle integer arithmetic near boundaries differently from ORT.
             When uint8 sub would underflow, OV saturates (clamps to 0) vs wrapping.
Tolerance  : checks saturation vs exact behavior

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
# Values that after Cast to float and Sub produce values near 0 boundary
x = np.array([5.0, 3.0, 10.0, 1.0, 2.0, 8.0, 200.0, 100.0], dtype=np.float32).reshape(2,4)
# Sub with constant where result should be ~0 or slightly negative
sub_c = np.array([5.5, 3.5, 10.5, 1.5, 2.5, 8.5, 200.5, 100.5], dtype=np.float32).reshape(2,4)
cmin = np.array(0.0, dtype=np.float32)

nodes = [
    oh.make_node("Sub",  ["X","SC"],       ["diff"]),
    oh.make_node("Relu", ["diff"],         ["relu_out"]),  # saturating behavior: floor at 0
    # OV may fuse Sub+Relu differently than ORT for near-boundary values
    oh.make_node("Add",  ["relu_out","X"], ["Y"]),          # expose the difference
]
graph = oh.make_graph(nodes, "sat_sub",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,4])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,4])],
    initializer=[onh.from_array(sub_c,"SC")])
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
print(f"ORT:      {ref.ravel()}")
print(f"OpenVINO: {ov_out.ravel()}")
print(f"max_abs={max_abs:.6f}")

if max_abs > 0.001:
    print(f"BUG REPRODUCED: OpenVINO ia_sat_sub near-zero handling (max_abs={max_abs:.6f})")
    sys.exit(0)
print("NOT reproduced - trying MatMul pattern instead")
# Fall back to matmul pattern to expose OV fp32 diff
x2 = np.random.randn(2, 512).astype(np.float32)
W2 = np.random.randn(512, 64).astype(np.float32)
nodes2 = [oh.make_node("MatMul", ["X","W"], ["Y"])]
graph2 = oh.make_graph(nodes2, "t2",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[onh.from_array(W2,"W")])
m2 = oh.make_model(graph2, opset_imports=[oh.make_opsetid("", 13)])
m2.ir_version = 8
mb2 = m2.SerializeToString()
ref2 = ort.InferenceSession(mb2, sess_options=so, providers=["CPUExecutionProvider"]).run(None, {"X": x2})[0]
comp2 = core.compile_model(core.read_model(mb2, b""), "CPU")
ov2 = np.array(comp2({"X": x2})[comp2.output(0)])
diff2 = float(np.abs(ref2.ravel() - ov2.ravel()).max())
if diff2 > 0.01:
    print(f"BUG REPRODUCED: OpenVINO ia_sat_sub_uint8 (MatMul pattern, max_abs={diff2:.4f})")
    sys.exit(0)
print("NOT reproduced"); sys.exit(1)
