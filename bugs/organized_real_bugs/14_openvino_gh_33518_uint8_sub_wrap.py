import numpy as np
import onnx
from onnx import TensorProto, helper
import sys

try:
    import openvino as ov
except ImportError:
    print("SKIP: openvino not installed")
    sys.exit(2)

a = np.array([5,   200, 250, 0], dtype=np.uint8)
b = np.array([10,  100,  10, 50], dtype=np.uint8)

ref_wrap = ((a.astype(np.int32) - b.astype(np.int32)) % 256).astype(np.uint8)
ref_sat  = np.clip(a.astype(np.int32) - b.astype(np.int32), 0, 255).astype(np.uint8)

A_vi = helper.make_tensor_value_info("A", TensorProto.UINT8, [4])
B_vi = helper.make_tensor_value_info("B", TensorProto.UINT8, [4])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.UINT8, None)
node = helper.make_node("Sub", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "g", [A_vi, B_vi], [Y_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
try:
    ov_m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
    out = ov_m({"A": a, "B": b})[ov_m.output(0)]
except Exception as e:
    print(f"OpenVINO error: {e}")
    sys.exit(2)

print(f"a        : {a}")
print(f"b        : {b}")
print(f"OV out   : {out}")
print(f"wrap ref : {ref_wrap}  (correct — ONNX modular)")
print(f"sat  ref : {ref_sat}   (wrong  — saturation)")

is_sat  = np.array_equal(out, ref_sat)
is_wrap = np.array_equal(out, ref_wrap)

print(f"\nMatches wrap (correct)? {is_wrap}")
print(f"Matches sat  (wrong)?   {is_sat}")

PASS = is_sat and not is_wrap
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — OV uint8 Sub saturates instead of wrapping")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
