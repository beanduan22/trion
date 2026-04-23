#!/usr/bin/env python3
"""
RC-11f · residual add+relu + 4D matmul; TS correct only

Single-case cluster (1 bug).  Representative = unique_0055.
Suspect backend(s): ['openvino', 'tensorflow', 'torch_compile', 'tvm', 'xla'].
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as oh, numpy_helper as onh

from _shared import locate_repo_root, run_backend, report  # noqa

locate_repo_root()

TOL = 1e-3
np.random.seed(25)
SHAPE = (1, 8, 8, 8)


def build_model():
    B, C, H, W = SHAPE; D = C

    Wq = np.random.randn(D, D).astype(np.float32) * 0.1
    Wv = np.random.randn(D, D).astype(np.float32) * 0.1
    w  = np.random.randn(D, D, 1, 1).astype(np.float32) * 0.1
    b  = np.zeros(D, np.float32)
    nodes = [
    oh.make_node("Add",     ["x", "x"], ["rs"]),
    oh.make_node("Relu",    ["rs"], ["rl"]),
    oh.make_node("MatMul",  ["rl", "Wq"], ["q"]),
    oh.make_node("Softmax", ["q"], ["sm"], axis=-1),
    oh.make_node("MatMul",  ["sm", "Wv"], ["mm"]),
    oh.make_node("Conv",    ["mm", "w", "b"], ["y"], kernel_shape=[1,1]),
    ]
    inits = [onh.from_array(Wq, "Wq"), onh.from_array(Wv, "Wv"),
    onh.from_array(w, "w"), onh.from_array(b, "b")]
    graph = oh.make_graph(
        nodes, "rc_11f_min",
        [oh.make_tensor_value_info("x", TP.FLOAT, list(SHAPE))],
        [oh.make_tensor_value_info("y", TP.FLOAT, [])],
        initializer=inits,
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def main():
    model = build_model()
    onnx.checker.check_model(model)
    inputs = {"x": np.random.randn(*SHAPE).astype(np.float32)}
    ref, err_ref = run_backend("onnxruntime", model, inputs)
    if err_ref:
        print(f"[setup] ORT failed: {err_ref}"); return 2
    outs = [(n, run_backend(n, model, inputs)[0]) for n in ['openvino', 'tensorflow', 'torch_compile', 'tvm', 'xla']]
    code, _ = report(expected_pair=("onnxruntime", ref),
                     suspect_pair=outs, outputs=None, tolerance=TOL)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
