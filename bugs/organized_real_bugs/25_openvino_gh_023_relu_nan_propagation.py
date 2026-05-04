import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import sys

try:
    import openvino as ov
except ImportError:
    print("SKIP: openvino not installed")
    sys.exit(2)

x = np.array([float('nan'), -1.0, 0.0, 1.0, float('inf')], dtype=np.float32)

X_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("Relu", ["X"], ["Y"])
graph = helper.make_graph([node], "g", [X_vi], [Y_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 6)])

model_bytes = model.SerializeToString()
mb = model_bytes

sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

core = ov.Core()
try:
    ov_m = core.compile_model(core.read_model(model_bytes, b""), "CPU")
    ov_out = ov_m({"X": x})[ov_m.output(0)]
except Exception as e:
    print(f"OpenVINO error: {e}")
    sys.exit(2)

print(f"Input  : {x}")
print(f"ORT    : {ort_out}  ← NaN propagated (correct)")
print(f"OV     : {ov_out}")

ort_nan = bool(np.isnan(ort_out[0]))
ov_nan  = bool(np.isnan(ov_out[0]))

print(f"\nInput[0] is NaN: True")
print(f"ORT[0]  is NaN: {ort_nan}  (expected: True)")
print(f"OV[0]   is NaN: {ov_nan}  (expected: True)")

PASS = (not ov_nan) and ort_nan
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754)")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
