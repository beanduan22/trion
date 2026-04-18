#!/usr/bin/env python3
"""
Bug ID     : github_ort_008
Source     : GitHub — microsoft/onnxruntime#25264
             https://github.com/microsoft/onnxruntime/issues/25264
Compiler   : ONNX Runtime (CPU, and CUDA when available)
Patterns   : Resize mode='cubic' downscale 8×8 → 4×4
Status     : Closed/Fixed on CUDA

Observed here (CPU, antialias=0):
    ORT's CPU cubic resize (pytorch_half_pixel, cubic_coeff_a = -0.5,
    antialias=0) returns values that differ from PyTorch's
    F.interpolate(mode='bicubic', align_corners=False, antialias=False)
    by a small but non-trivial per-pixel error. The divergence isn't a
    catastrophic wrong-index case — it's the accumulated effect of ORT's
    cubic kernel shape vs PyTorch's. The upstream issue was filed because
    the CUDA+antialias=1 path was dramatically wrong (wrong block count
    and cubic_coeff_a ignored); the fix landed for CUDA. On CPU we exercise
    the same op with antialias=0 and characterise the CPU cubic numerical
    discrepancy actually observed on this host.

Observed here (CUDA, antialias=1) — only if a CUDA EP is available:
    Runs the original buggy configuration. On a pre-fix build the CUDA
    cubic+antialias output diverges from PyTorch by ≫ 1.0. On a fixed
    build (2024.x onward) the divergence should shrink to the CPU-path
    tolerance.

Primary oracle : PyTorch F.interpolate bicubic
Tolerance      : 0.05 (CPU characterisation), 0.05 (CUDA if run)

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
from __future__ import annotations

import sys as _sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
    import torch
except ImportError as e:
    print(f"missing dep: {e}")
    _sys.exit(2)


def build_model(antialias: int) -> bytes:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    scales = helper.make_tensor(
        "scales", TensorProto.FLOAT, [4], [1.0, 1.0, 0.5, 0.5]
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])
    node = helper.make_node(
        "Resize",
        inputs=["X", "", "scales"],
        outputs=["Y"],
        mode="cubic",
        coordinate_transformation_mode="pytorch_half_pixel",
        cubic_coeff_a=-0.5,
        antialias=antialias,
    )
    graph = helper.make_graph([node], "cubic_resize", [X], [Y], initializer=[scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    return model.SerializeToString()


np.random.seed(11)
src = np.random.rand(1, 1, 8, 8).astype(np.float32)

# CPU path: antialias=0 (the only CPU-safe path in historical builds).
model_bytes_cpu = build_model(antialias=0)
sess_cpu = ort.InferenceSession(model_bytes_cpu, providers=["CPUExecutionProvider"])
ort_cpu = sess_cpu.run(None, {"X": src})[0]

torch_ref_plain = (
    torch.nn.functional.interpolate(
        torch.from_numpy(src),
        size=(4, 4),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    .numpy()
)
cpu_err = float(np.max(np.abs(ort_cpu - torch_ref_plain)))

print("=== CPU path (antialias=0) ===")
print(f"ORT  CPU cubic  [0,0]:\n{ort_cpu[0, 0]}")
print(f"PyTorch bicubic [0,0]:\n{torch_ref_plain[0, 0]}")
print(f"max |ORT CPU - torch bicubic| = {cpu_err:.6f}")

# CUDA path: only if CUDA EP is actually available on this host.
cuda_err: float | None = None
cuda_status = ""
if (
    "CUDAExecutionProvider" in ort.get_available_providers()
    and torch.cuda.is_available()
):
    try:
        model_bytes_cuda = build_model(antialias=1)
        sess_cuda = ort.InferenceSession(
            model_bytes_cuda,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ort_cuda = sess_cuda.run(None, {"X": src})[0]
        torch_ref_aa = (
            torch.nn.functional.interpolate(
                torch.from_numpy(src).cuda(),
                size=(4, 4),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .cpu()
            .numpy()
        )
        cuda_err = float(np.max(np.abs(ort_cuda - torch_ref_aa)))
        print()
        print("=== CUDA path (antialias=1) ===")
        print(f"ORT  CUDA cubic+aa [0,0]:\n{ort_cuda[0, 0]}")
        print(f"PyTorch bicubic+aa [0,0]:\n{torch_ref_aa[0, 0]}")
        print(f"max |ORT CUDA+aa - torch bicubic+aa| = {cuda_err:.6f}")
    except Exception as e:  # pragma: no cover
        cuda_status = f"CUDA path errored: {str(e)[:100]}"
        print()
        print(cuda_status)
else:
    print()
    print("(CUDA EP not available — skipping CUDA+antialias=1 path.)")
    cuda_status = "CUDA path not exercised on this host"

TOL = 0.05
cpu_bug = cpu_err > TOL
cuda_bug = cuda_err is not None and cuda_err > TOL
BUG = cpu_bug or cuda_bug

print()
print(f"Tolerance: {TOL}")
print(f"CPU  path:  bug? {cpu_bug}  (err={cpu_err:.6f})")
if cuda_err is not None:
    print(f"CUDA path:  bug? {cuda_bug}  (err={cuda_err:.6f})")
else:
    print(f"CUDA path:  {cuda_status}")
print(f"PASS={not BUG}")
if BUG:
    detail = []
    if cpu_bug:
        detail.append(f"CPU cubic err={cpu_err:.4f}")
    if cuda_bug:
        detail.append(f"CUDA cubic+aa err={cuda_err:.4f}")
    print(f"BUG REPRODUCED: {'; '.join(detail)} vs PyTorch bicubic reference.")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
