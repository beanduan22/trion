import signal
import subprocess
import sys

code = r"""
import numpy as np, onnxruntime as ort
from onnx import helper, TensorProto
g = helper.make_graph([helper.make_node("Mod", ["A", "B"], ["Y"], fmod=0)], "g",
    [helper.make_tensor_value_info("A", TensorProto.INT32, [2]), helper.make_tensor_value_info("B", TensorProto.INT32, [2])],
    [helper.make_tensor_value_info("Y", TensorProto.INT32, [2])])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {
    "A": np.array([7, 13], dtype=np.int32), "B": np.array([0, 0], dtype=np.int32)})
"""
proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
bug = proc.returncode == -signal.SIGFPE or "division by zero" in proc.stderr.lower()
print("BUG REPRODUCED" if bug else "not reproduced")
sys.exit(0 if bug else 1)
