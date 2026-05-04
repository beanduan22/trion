import sys
import numpy as np
import onnxruntime as ort
import openvino as ov
from onnx import TensorProto, helper

x = np.random.randn(1, 1, 8, 8).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], pads=[3, 3, 3, 3], strides=[1, 1])],
    "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 12, 12])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
mb = m.SerializeToString()
try:
    ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, {"x": x})
    print("not reproduced")
    sys.exit(1)
except Exception:
    out = ov.Core().compile_model(ov.Core().read_model(mb, b""), "CPU")({"x": x})
    bug = list(out.values())[0].size > 0
    print("BUG REPRODUCED" if bug else "not reproduced")
    sys.exit(0 if bug else 1)
