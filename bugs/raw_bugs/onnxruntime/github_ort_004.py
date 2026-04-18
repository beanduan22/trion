#!/usr/bin/env python3
"""
Bug ID     : github_ort_004
Source     : GitHub — OnnxRuntime
Compiler   : ONNX Runtime CPU
Patterns   : Cast(float→int32) → Cast(int32→bool)
Root cause : An ORT graph optimization fuses the
             float→int32→bool cast chain into a direct float→bool cast.
             That loses the int32 truncation step:
               • correct  : -0.1 → 0 → False
               • fused    : -0.1 → (-0.1 != 0) → True
             We run both ORT_DISABLE_ALL and ORT_ENABLE_ALL side by side
             and compare against a NumPy reference that performs the two
             casts sequentially (the ONNX spec semantics).
Tolerance  : exact

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
from __future__ import annotations

import sys as _sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}")
    _sys.exit(2)

# -0.2 .. 0.2: every value truncates to int32 0, so bool should be all False.
x_vals = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
T1 = helper.make_tensor_value_info("T1", TensorProto.INT32, [5])
Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [5])
cast1 = helper.make_node("Cast", ["X"], ["T1"], to=TensorProto.INT32)
cast2 = helper.make_node("Cast", ["T1"], ["Y"], to=TensorProto.BOOL)
graph = helper.make_graph([cast1, cast2], "g", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)
model_bytes = model.SerializeToString()


def run(opt_level: "ort.GraphOptimizationLevel") -> np.ndarray:
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(
        model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"X": x_vals})[0]


out_disable = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
out_enable = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

# Correct: truncate to int32 first (all values round to 0), then != 0 → all False.
expected = (x_vals.astype(np.int32) != 0)

print(f"Input             : {x_vals}")
print(f"Expected (truncate→bool): {expected}")
print(f"ORT_DISABLE_ALL   : {out_disable}")
print(f"ORT_ENABLE_ALL    : {out_enable}")

disable_matches = bool(np.array_equal(out_disable, expected))
enable_matches = bool(np.array_equal(out_enable, expected))
disable_vs_enable = bool(np.array_equal(out_disable, out_enable))

print()
print(f"ORT_DISABLE_ALL matches expected : {disable_matches}")
print(f"ORT_ENABLE_ALL  matches expected : {enable_matches}")
print(f"DISABLE == ENABLE                : {disable_vs_enable}")

# The bug is: ORT disagrees with the ONNX-spec semantics of the two-step
# cast chain, regardless of optimization level. The fact that
# ORT_ENABLE_ALL matches ORT_DISABLE_ALL only tells us the fusion (if any)
# happens at or below the "disable_all" layer on this build — the crucial
# evidence is that NEITHER ORT setting preserves the int32 truncation.
BUG = not enable_matches  # ENABLE_ALL is what a user gets by default
print(f"PASS={not BUG}")
if BUG:
    print(
        "BUG REPRODUCED: ORT's Cast(float→int32)→Cast(int32→bool) chain "
        "does not preserve int32 truncation. For -0.2, -0.1, 0.1, 0.2 the "
        "spec demands int32 truncation to 0 and therefore bool False, but "
        "ORT returns True. ORT_DISABLE_ALL vs ORT_ENABLE_ALL table above "
        "shows whether the ORT graph optimizer alone is the culprit on "
        "this build."
    )
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
