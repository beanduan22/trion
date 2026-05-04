import numpy as np
import onnx
from onnx import TensorProto, helper
import sys

try:
    import openvino as ov
except ImportError:
    print("SKIP: openvino not installed")
    sys.exit(2)

a = np.array([100,  127, -128, -100], dtype=np.int8)
b = np.array([100,   10,  -10,  -50], dtype=np.int8)

ref_wrap = (a.astype(np.int16) + b.astype(np.int16)).astype(np.int8)
ref_sat  = np.clip(a.astype(np.int16) + b.astype(np.int16), -128, 127).astype(np.int8)

A_vi = helper.make_tensor_value_info("A", TensorProto.INT8, [4])
B_vi = helper.make_tensor_value_info("B", TensorProto.INT8, [4])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.INT8, None)
node = helper.make_node("Add", ["A", "B"], ["Y"])
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
print(f"a+b(i16) : {(a.astype(np.int16)+b.astype(np.int16))}")
print(f"OV out   : {out}")
print(f"wrap ref : {ref_wrap}  (correct — two's complement)")
print(f"sat  ref : {ref_sat}   (wrong  — saturation)")

is_sat  = np.array_equal(out, ref_sat)
is_wrap = np.array_equal(out, ref_wrap)

PASS = is_sat and not is_wrap
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — OV int8 Add saturates instead of two's-complement wrapping")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
