import sys
import numpy as np
import onnxruntime as ort
import openvino as ov
from onnx import TensorProto, helper, numpy_helper

params = np.arange(1, 13, dtype=np.float32)
idx = np.array([12], dtype=np.int64)
g = helper.make_graph(
    [helper.make_node("Gather", ["params", "idx"], ["Y"], axis=0)],
    "g",
    [helper.make_tensor_value_info("params", TensorProto.FLOAT, [12])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    initializer=[numpy_helper.from_array(idx, "idx")],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
mb = m.SerializeToString()
try:
    ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, {"params": params})
    print("not reproduced")
    sys.exit(1)
except Exception:
    out = ov.Core().compile_model(ov.Core().read_model(mb, b""), "CPU")({"params": params})
    bug = float(list(out.values())[0][0]) == 0.0
    print("BUG REPRODUCED" if bug else "not reproduced")
    sys.exit(0 if bug else 1)
