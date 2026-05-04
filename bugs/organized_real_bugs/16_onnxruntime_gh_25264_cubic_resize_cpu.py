import sys
import numpy as np
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

x = np.random.RandomState(11).rand(1, 1, 8, 8).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("Resize", ["X", "", "scales"], ["Y"], mode="cubic", coordinate_transformation_mode="pytorch_half_pixel", cubic_coeff_a=-0.5, antialias=0)],
    "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])],
    initializer=[helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 0.5, 0.5])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
m.ir_version = 9
ort_out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
ref = torch.nn.functional.interpolate(torch.from_numpy(x), size=(4, 4), mode="bicubic", align_corners=False, antialias=False).numpy()
bug = float(np.max(np.abs(ort_out - ref))) > 0.05
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
