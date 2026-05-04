import numpy as np
import onnx
from onnx import TensorProto, helper
import sys

try:
    import openvino as ov
except ImportError:
    print("SKIP: openvino not installed")
    sys.exit(2)

try:
    import torch
    x = np.array([[100.0, 88.0, 50.0], [200.0, -10.0, 1.0]], dtype=np.float32)
    ref = torch.logsumexp(torch.tensor(x), dim=1).numpy()
except ImportError:
    x = np.array([[100.0, 88.0, 50.0], [200.0, -10.0, 1.0]], dtype=np.float32)
    m = x.max(axis=1, keepdims=True)
    ref = (m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))).squeeze(1)

X_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("ReduceLogSumExp", ["X"], ["Y"], axes=[1], keepdims=0)
graph = helper.make_graph([node], "g", [X_vi], [Y_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

core = ov.Core()
try:
    ov_m = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
    out = ov_m({"X": x})[ov_m.output(0)]
except Exception as e:
    print(f"OpenVINO error: {e}")
    sys.exit(2)

print(f"Input x  : {x}")
print(f"OV out   : {out}")
print(f"Ref      : {ref}")
has_inf = bool(np.any(np.isinf(out)))
diff     = float(np.max(np.abs(out - ref))) if not has_inf else float('inf')
print(f"has_inf  : {has_inf}")
print(f"max_diff : {diff:.4f}")

PASS = has_inf or diff > 0.1
print(f"\nPASS={PASS}")
if PASS:
    print("BUG REPRODUCED — OV ReduceLogSumExp overflows for inputs ≥ 88.7 (fp32 exp limit)")
    sys.exit(0)
print("not reproduced")
sys.exit(1)
