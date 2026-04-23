#!/usr/bin/env python3
"""
RC-03 · fp16 cast-roundtrip + fused softmax-stability chain.

Cluster: 5 unique bugs.
Representative source: unique_0010 (campaign bug #233).

Trigger pattern:
    x -> Cast(fp32->fp16) -> Cast(fp16->fp32) -> manual softmax
         (Neg -> Sub(max) -> Exp -> ReduceSum -> Div) -> LayerNorm

Reference cluster: pytorch_eager, onnxruntime, torchscript, tvm.
Suspect cluster  : openvino, tensorflow, torch_compile, xla.

The four suspect backends see the Cast fp32->fp16->fp32 as dead code and
eliminate it, recovering fp32 precision.  The ONNX spec, ORT, TS, and TVM
keep the roundtrip and its precision loss — that loss is the "correct"
reference here.  rel_l2 ~ 0.06 – 1.0 depending on the softmax scale.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

B, N, D = 1, 4, 16
TOL = 1e-3
np.random.seed(2)


def build_model():
    axes_last = np.array([-1], dtype=np.int64)
    nodes = [
        oh.make_node("Cast",  ["x"],   ["x16"], to=TP.FLOAT16),
        oh.make_node("Cast",  ["x16"], ["xf"],  to=TP.FLOAT),
        oh.make_node("Neg",   ["xf"],  ["nx"]),
        oh.make_node("ReduceMax", ["nx", "axes_last"], ["m"], keepdims=1),
        oh.make_node("Sub",   ["nx", "m"],  ["c"]),
        oh.make_node("Exp",   ["c"],        ["e"]),
        oh.make_node("ReduceSum", ["e", "axes_last"], ["s"], keepdims=1),
        oh.make_node("Div",   ["e", "s"],   ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "rc03_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, [B, N, D])],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=[onh.from_array(axes_last, "axes_last")],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": (np.random.randn(B, N, D).astype(np.float32) * 12.0)}  # big range => fp16 loss

    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref: print(f"[setup] ORT failed: {err_ref}"); return 2

    outs = [(name, run_backend(name, model, inputs)[0])
            for name in ["openvino", "tensorflow", "torch_compile", "xla"]]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
