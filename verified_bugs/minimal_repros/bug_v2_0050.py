#!/usr/bin/env python3
"""
Bug v2-0050 — ORT SUSPECT: CumSum(last axis) + cast_roundtrip(f32->f16->f32) + RMSNorm + TopK
Compiler : OnnxRuntime
Pattern  : CumSum -> Cast(f16) -> Cast(f32) -> RMSNorm -> TopK
Suspect  : Divergence vs pytorch_eager reference; ORT_ENABLE_ALL vs ORT_DISABLE_ALL
           results are close — the bug may be a cast precision difference.
Status   : SUSPECT — shown for completeness; PASS means ORT self-consistent.
Note     : Cast to=10 is FLOAT16; Cast to=1 is FLOAT.

Run: python bug_v2_0050.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

np.random.seed(9)

N, D = 1, 16

def make_model():
    nodes = [
        # CumSum on last axis (axis=1 for [1,D])
        helper.make_node("CumSum", ["x", "cs_ax"], ["cs_out"]),
        # Cast roundtrip: f32 -> f16 -> f32
        helper.make_node("Cast", ["cs_out"], ["cast_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["cast_f16"], ["cast_f32"], to=TensorProto.FLOAT),
        # RMSNorm: x^2 -> ReduceMean(last) -> Add(eps) -> Sqrt -> Div
        helper.make_node("Mul", ["cast_f32", "cast_f32"], ["rms_sq"]),
        helper.make_node("ReduceMean", ["rms_sq"], ["rms_mean"], axes=[1], keepdims=0),
        helper.make_node("Add", ["rms_mean", "rms_eps"], ["rms_add"]),
        helper.make_node("Sqrt", ["rms_add"], ["rms_sqrt"]),
        helper.make_node("Div", ["cast_f32", "rms_sqrt"], ["rms_out"]),
        # TopK (k=1, largest=True, sorted=True, axis=-1)
        helper.make_node("TopK", ["rms_out", "tk_k"], ["tk_vals", "tk_idx"],
                         axis=-1, largest=1, sorted=1),
        # Tile to restore shape and compute abs-relu
        helper.make_node("Tile", ["tk_vals", "tk_rep"], ["tk_tiled"]),
        helper.make_node("Add", ["tk_tiled", "tk_tiled"], ["ars_ad"]),
        helper.make_node("Relu", ["ars_ad"], ["ars_rl"]),
        helper.make_node("Sub", ["ars_rl", "tk_tiled"], ["out"]),
    ]

    initializers = [
        numpy_helper.from_array(np.array(1, dtype=np.int64), "cs_ax"),
        numpy_helper.from_array(np.array([1e-6], dtype=np.float32), "rms_eps"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "tk_k"),
        numpy_helper.from_array(np.array([1, D], dtype=np.int64), "tk_rep"),
    ]

    graph = helper.make_graph(
        nodes, "bug_v2_0050",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, D])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [N, D])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()

model_bytes = make_model()
x_in = np.random.randn(N, D).astype(np.float32)

def run(opt_level):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x_in})[0]

out_all = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
out_dis = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

max_diff = float(np.max(np.abs(out_all - out_dis)))
passed = max_diff < 0.01
print(f"uid=0050 SUSPECT CumSum + cast_f32_f16_f32 + RMSNorm + TopK")
print(f"  ORT_ENABLE_ALL vs ORT_DISABLE_ALL max_diff={max_diff:.6f}")
print(f"  (Bug was vs pytorch_eager reference, not ORT self-inconsistency)")
print(f"PASS={passed}")
