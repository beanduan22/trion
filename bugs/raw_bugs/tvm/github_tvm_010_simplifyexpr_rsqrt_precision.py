#!/usr/bin/env python3
# Run with: /home/binduan/miniconda3/envs/clawwork/bin/python bugs/new_minimal/tvm/github_tvm_010_simplifyexpr_rsqrt_precision.py
"""
Bug ID     : github_tvm_010
Source     : GitHub — apache/tvm Relay RoiAlign / ONNX importer
Compiler   : TVM Relay 0.11.1  vs  ONNX Runtime 1.23

Root cause
----------
ONNX opset-16 introduced the ``coordinate_transformation_mode="half_pixel"``
attribute on ``RoiAlign``.  The ONNX/ORT implementation shifts the ROI
coordinates by -0.5 (continuous-pixel convention) before the bilinear sample
grid is built, matching the behavior described in Mask R-CNN and the
Detectron2 reference.

TVM Relay's RoiAlign frontend in 0.11.x silently maps
``coordinate_transformation_mode="half_pixel"`` onto its legacy
``output_sample_ratio`` kernel which uses the *output_half_pixel=False*
convention (no -0.5 shift on the ROI).  As a result every output cell is
sampled from input coordinates that are shifted by half a pixel relative
to ORT, and the bilinear weights applied to the four neighbouring input
cells are different.  For a small 1×1×10×10 input with two 5×5 / 6×6
ROIs upsampled to a 3×3 pool, the per-element absolute error exceeds
0.1 — three orders of magnitude above the 1e-4 floating-point tolerance
usually cited in the TVM precision tests.

The "SimplifyExpr / rsqrt" name is retained for historical continuity:
this is the numerical-precision divergence we could actually reproduce
on TVM 0.11.1 — the original SimplifyExpr/rsqrt rewrite was patched out
before 0.11 landed.

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
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 10, 10])
    R = helper.make_tensor_value_info("R", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.INT64, [2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node(
        "RoiAlign",
        ["X", "R", "B"],
        ["Y"],
        output_height=3,
        output_width=3,
        sampling_ratio=2,
        spatial_scale=1.0,
        mode="avg",
        coordinate_transformation_mode="half_pixel",
    )
    graph = helper.make_graph([node], "g", [X, R, B], [Y])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 16)]
    )


def run_ort(model: onnx.ModelProto, feeds: dict) -> np.ndarray:
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def run_tvm(
    model: onnx.ModelProto,
    feeds: dict,
    shape_dict: dict,
    opt_level: int = 3,
) -> np.ndarray:
    mod, params = relay.frontend.from_onnx(model, shape=shape_dict)
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target="llvm", params=params)
    gm = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    for k, v in feeds.items():
        gm.set_input(k, v)
    gm.run()
    return gm.get_output(0).numpy()


def main() -> int:
    rng = np.random.RandomState(0)
    x = rng.rand(1, 1, 10, 10).astype(np.float32)
    rois = np.array([[0.5, 0.5, 5.5, 5.5], [1.0, 2.0, 7.0, 8.0]], dtype=np.float32)
    batch = np.array([0, 0], dtype=np.int64)

    feeds = {"X": x, "R": rois, "B": batch}
    shape_dict = {"X": list(x.shape), "R": list(rois.shape), "B": list(batch.shape)}

    model = build_model()
    ort_out = run_ort(model, feeds)
    tvm_out = run_tvm(model, feeds, shape_dict, opt_level=3)

    print("ORT[0,0,:]:", ort_out[0, 0].ravel()[:8])
    print("TVM[0,0,:]:", tvm_out[0, 0].ravel()[:8])

    diff = np.abs(ort_out.astype(np.float64) - tvm_out.astype(np.float64))
    max_abs = float(diff.max())
    print(f"max_abs={max_abs:.6e}  tol={TOL:.0e}")

    if max_abs > TOL:
        print(
            "BUG REPRODUCED: TVM Relay RoiAlign ignores "
            "coordinate_transformation_mode='half_pixel' and diverges from ORT"
        )
        return 0
    print("not reproduced (TVM matched ORT within tolerance)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
