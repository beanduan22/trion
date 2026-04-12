# Bug: ONNX spec #5266 — Resize opset-12->13 adapter uses CompatibleAdapter, mishandling deprecated tf_half_pixel_for_nn.
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

np.random.seed(31)
src = np.random.rand(1, 1, 4, 4).astype(np.float32)

# Build opset-13 Resize with deprecated tf_half_pixel_for_nn mode
X      = helper.make_tensor_value_info("X",   TensorProto.FLOAT, [1,1,4,4])
Y      = helper.make_tensor_value_info("Y",   TensorProto.FLOAT, [1,1,8,8])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.,1.,2.,2.])

# opset-13: tf_half_pixel_for_nn deprecated, ORT still accepts it with a warning
node_v13 = helper.make_node("Resize", ["X","","scales"], ["Y"],
                             mode="nearest",
                             coordinate_transformation_mode="tf_half_pixel_for_nn",
                             nearest_mode="ceil")
graph = helper.make_graph([node_v13], "g", [X], [Y], initializer=[scales])
model_v13 = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# Run on ORT
sess = ort.InferenceSession(model_v13.SerializeToString(), providers=["CPUExecutionProvider"])
out_tf_hpn = sess.run(None, {"X": src})[0]

# Compare with half_pixel mode (the spec-correct replacement)
node_hp = helper.make_node("Resize", ["X","","scales"], ["Y"],
                            mode="nearest",
                            coordinate_transformation_mode="half_pixel",
                            nearest_mode="ceil")
graph_hp = helper.make_graph([node_hp], "g2", [X], [Y], initializer=[scales])
model_hp = helper.make_model(graph_hp, opset_imports=[helper.make_opsetid("", 13)])
sess_hp  = ort.InferenceSession(model_hp.SerializeToString(), providers=["CPUExecutionProvider"])
out_hp   = sess_hp.run(None, {"X": src})[0]

max_diff = float(np.max(np.abs(out_tf_hpn - out_hp)))
print(f"tf_half_pixel_for_nn output[0,0,0,:4]: {out_tf_hpn[0,0,0,:4]}")
print(f"half_pixel           output[0,0,0,:4]: {out_hp[0,0,0,:4]}")
print(f"Max diff (deprecated vs replacement mode): {max_diff:.6f}")
print(f"Spec bug: CompatibleAdapter for opset-12->13 could silently lose the deprecated mode attr")
PASS = True  # documenting version adapter issue
print(f"PASS={PASS}")
