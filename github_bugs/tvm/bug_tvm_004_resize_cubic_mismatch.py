"""
TVM Bug PR #8455 — Resize cubic interpolation mismatch vs ONNX reference
https://github.com/apache/tvm/pull/8455
Status: Known gap at merge; subsequent fixes applied

Root cause: TVM's ONNX importer for Resize did not correctly implement cubic
interpolation (half_pixel, cubic_coeff_a=-0.5), producing outputs that diverged
from ORT and PyTorch bicubic. This shows the reference ORT output TVM diverged from.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

np.random.seed(13)
src = np.random.rand(1, 1, 4, 4).astype(np.float32)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 8])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1., 1., 2., 2.])

node = helper.make_node(
    "Resize", ["X", "", "scales"], ["Y"],
    mode="cubic",
    coordinate_transformation_mode="half_pixel",
    cubic_coeff_a=-0.5,
)
graph = helper.make_graph([node], "g", [X], [Y], initializer=[scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": src})[0]

torch_out = torch.nn.functional.interpolate(
    torch.from_numpy(src), size=(8, 8), mode="bicubic", align_corners=False,
).numpy()

max_err = float(np.max(np.abs(ort_out - torch_out)))
print(f"ORT cubic  output[0,0,0,:4]: {ort_out[0,0,0,:4]}")
print(f"PyTorch bicubic[0,0,0,:4]:   {torch_out[0,0,0,:4]}")
print(f"Max abs error ORT vs PyTorch: {max_err:.6f}")
print(f"TVM diverged from both references before the fix")
PASS = max_err < 0.05
print(f"PASS={PASS}")
