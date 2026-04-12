import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

# 2-channel 4x4 input, block_size=2 -> output [1, 8, 2, 2]
x_nchw = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 2, 2])

node  = helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
graph = helper.make_graph([node], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

sess    = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x_nchw})[0]  # DCR ordering

# PyTorch pixel_unshuffle: CRD ordering
torch_out = torch.nn.functional.pixel_unshuffle(
    torch.from_numpy(x_nchw), downscale_factor=2).numpy()

max_diff = float(np.max(np.abs(ort_out - torch_out)))
print(f"Input NCHW: {x_nchw.shape}, block_size=2 -> output {ort_out.shape}")
print(f"ONNX SpaceToDepth (DCR) [0,:4,0,0]: {ort_out[0,:4,0,0]}")
print(f"PyTorch pixel_unshuffle (CRD)[0,:4,0,0]: {torch_out[0,:4,0,0]}")
print(f"Max diff DCR vs CRD: {max_diff:.4f}")
print(f"TF bug: multi-channel space_to_depth used wrong channel ordering")
print(f"Ordering mismatch confirmed (diff > 0): {max_diff > 0}")
PASS = max_diff > 0  # confirms the DCR vs CRD ordering difference exists
print(f"PASS={PASS}")
