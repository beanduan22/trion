#!/usr/bin/env python3
# Run with: /home/binduan/miniconda3/envs/clawwork/bin/python bugs/new_minimal/tvm/github_tvm_004.py
"""
Bug ID     : github_tvm_004
Source     : GitHub — apache/tvm Relay ONNX Resize importer
Compiler   : TVM Relay 0.11.1  vs  ONNX Runtime 1.23

Root cause
----------
TVM Relay's ONNX importer for the ``Resize`` operator mishandles
``mode="nearest"`` with ``coordinate_transformation_mode="half_pixel"`` +
``nearest_mode="round_prefer_floor"``.  The ONNX spec maps an output pixel
``x_out`` back to the input via ``x_in = (x_out + 0.5) / scale - 0.5``;
with ``round_prefer_floor`` a half-integer result should round down.
Relay instead computes the source coordinate without the ``-0.5`` shift
(effectively ``asymmetric``-style) and then applies ``floor`` rounding,
producing source indices that are off by one to two rows/cols for the
very first output coordinate of each axis.

For a 3x3 input upsampled by 1.5x to a 4x4 output, ORT returns

    [[0 0 1 2]
     [0 0 1 2]
     [3 3 4 5]
     [6 6 7 8]]

but TVM Relay returns

    [[0 1 1 2]
     [3 4 4 5]
     [3 4 4 5]
     [6 7 7 8]]

a per-pixel absolute error of up to 4.0 (indices read from wrong row).

Exit codes
----------
0 = BUG REPRODUCED (TVM differs from ORT by > TOL)
1 = TVM matched ORT (no bug)
2 = missing TVM or onnxruntime import
"""
from __future__ import annotations

import sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
except ImportError as e:
    print(f"missing dependency: {e}")
    sys.exit(2)


TOL: float = 1e-4


def build_model() -> onnx.ModelProto:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3])
    # Output shape unspecified (let backend infer; ORT/TVM both produce 4x4).
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    scales = helper.make_tensor(
        "scales", TensorProto.FLOAT, [4], [1.0, 1.0, 1.5, 1.5]
    )
    node = helper.make_node(
        "Resize",
        ["X", "", "scales"],
        ["Y"],
        mode="nearest",
        coordinate_transformation_mode="half_pixel",
        nearest_mode="round_prefer_floor",
    )
    graph = helper.make_graph([node], "g", [X], [Y], initializer=[scales])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    return model


def run_ort(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"X": x})[0]


def run_tvm(model: onnx.ModelProto, x: np.ndarray, opt_level: int = 3) -> np.ndarray:
    mod, params = relay.frontend.from_onnx(model, shape={"X": list(x.shape)})
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target="llvm", params=params)
    gm = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    gm.set_input("X", x)
    gm.run()
    return gm.get_output(0).numpy()


def main() -> int:
    x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)

    model = build_model()
    ort_out = run_ort(model, x)
    tvm_out = run_tvm(model, x, opt_level=3)

    print("input:")
    print(x[0, 0])
    print("ORT output [0,0]:")
    print(ort_out[0, 0])
    print("TVM output [0,0]:")
    print(tvm_out[0, 0])

    diff = np.abs(ort_out.astype(np.float64) - tvm_out.astype(np.float64))
    max_abs = float(diff.max())
    print(f"max_abs={max_abs:.6e}  tol={TOL:.0e}")

    if max_abs > TOL:
        print(
            "BUG REPRODUCED: TVM Relay Resize(nearest, half_pixel, "
            "round_prefer_floor) diverges from ORT"
        )
        return 0
    print("not reproduced (TVM matched ORT within tolerance)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
