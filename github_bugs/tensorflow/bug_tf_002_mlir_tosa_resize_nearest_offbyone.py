import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: TF MLIR/TOSA nearest resize shifted rows by 1 with half_pixel_centers=True (issue #62386).
src = np.array([[[[0.], [1.], [2.], [3.]]]], dtype=np.float32)

X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 1])
Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 1])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1., 1., 2., 1.])

n_half = helper.make_node("Resize", ["X","","scales"], ["Y"],
    mode="nearest", coordinate_transformation_mode="half_pixel", nearest_mode="floor")
g_half = helper.make_graph([n_half], "g", [X], [Y], initializer=[scales])
m_half = helper.make_model(g_half, opset_imports=[helper.make_opsetid("", 13)])
sess_half = ort.InferenceSession(m_half.SerializeToString(), providers=["CPUExecutionProvider"])

n_asym = helper.make_node("Resize", ["X","","scales"], ["Y"],
    mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
g_asym = helper.make_graph([n_asym], "g", [X], [Y], initializer=[scales])
m_asym = helper.make_model(g_asym, opset_imports=[helper.make_opsetid("", 13)])
sess_asym = ort.InferenceSession(m_asym.SerializeToString(), providers=["CPUExecutionProvider"])

out_half = sess_half.run(None, {"X": src})[0].flatten()
out_asym = sess_asym.run(None, {"X": src})[0].flatten()

# Asymmetric (no half_pixel offset): [0,0,1,1,2,2,3,3]
expected_asym = np.array([0., 0., 1., 1., 2., 2., 3., 3.])
# Half_pixel: samples at 0.25,0.75,1.25,... -> floor -> [0,0,1,1,2,2,3,3] same here
err_asym = float(np.max(np.abs(out_asym - expected_asym)))
err_half = float(np.max(np.abs(out_half - expected_asym)))

print(f"asymmetric output:          {out_asym}")
print(f"half_pixel output:          {out_half}")
print(f"Expected (both):            {expected_asym}")
print(f"Max err asymmetric: {err_asym:.4f}, half_pixel: {err_half:.4f}")
print(f"TOSA bug: wrong scale/offset shifted row index by 1 in compiled (not eager) path")
PASS = err_asym < 0.5 and err_half < 0.5
print(f"PASS={PASS}")
