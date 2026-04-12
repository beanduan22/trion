import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Bug: ORT_ENABLE_ALL optimizer strips float->int32->bool to float->bool.
# float->int32 truncates: -0.1 -> 0 -> False
# float->bool directly: -0.1 != 0 -> True  (wrong)
x_vals = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32)

X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [5])
T1 = helper.make_tensor_value_info("T1", TensorProto.INT32, [5])
Y  = helper.make_tensor_value_info("Y",  TensorProto.BOOL,  [5])
cast1 = helper.make_node("Cast", ["X"],  ["T1"], to=TensorProto.INT32)
cast2 = helper.make_node("Cast", ["T1"], ["Y"],  to=TensorProto.BOOL)
graph = helper.make_graph([cast1, cast2], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(model.SerializeToString(), sess_options=so,
                            providers=["CPUExecutionProvider"])
out = sess.run(None, {"X": x_vals})[0]

# Correct: truncate to int32 first (all values round to 0), then != 0 -> all False
expected = (x_vals.astype(np.int32) != 0)
print(f"Input:    {x_vals}")
print(f"Expected: {expected}")
print(f"ORT out:  {out}")
PASS = np.array_equal(out, expected)
print(f"PASS={PASS}")
