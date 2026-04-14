# 14 Live Bugs (Still Reproduce)

Verified 2026-04-13 on: ORT 1.24.4, PyTorch 2.9.1, ONNX 1.21.0, JAX 0.9.2+CUDA, OpenVINO 2026.0.

```bash
# Run all
for f in bugs/live/*.py; do
  python "$f" 2>/dev/null && echo "BUG: $f" || echo "OK: $f"
done
# Exit 0 = BUG REPRODUCED | Exit 1 = not reproduced | Exit 2 = missing deps
```

## Summary

| # | Bug ID | Compiler | max_diff | Root Cause |
|---|---|---|---|---|
| 1 | [cross_onnx2torch_cumsum](cross_onnx2torch_cumsum.py) | onnx2torch | **4.50** | CumSum axis.item() causes torch.compile graph break; wrong cumulative sum output |
| 2 | [cross_onnx2torch_resize_nearest_floor](cross_onnx2torch_resize_nearest_floor.py) | onnx2torch | **4.79** | onnx2torch maps nearest floor+half_pixel to wrong PyTorch mode; pixel selection diverges |
| 3 | [cross_onnx2torch_resize_nearest_ceil](cross_onnx2torch_resize_nearest_ceil.py) | onnx2torch | **3.00** | onnx2torch uses PyTorch floor nearest_mode instead of ONNX ceil; pixel selection wrong |
| 4 | [github_tensorflow_002](github_tensorflow_002.py) | TensorFlow XLA | **1.00** | MLIR/TOSA nearest resize shifts rows by 1 with half_pixel_centers=True |
| 5 | [cross_onnx2torch_resize_linear_asym](cross_onnx2torch_resize_linear_asym.py) | onnx2torch | **0.80** | ONNX asymmetric coordinate mode mapped to wrong PyTorch interpolation mode |
| 6 | [github_ort_003](github_ort_003.py) | OnnxRuntime 1.24.4 | **0.29** | Asymmetric bilinear resize gives 0.29 max error vs PyTorch reference |
| 7 | [github_tvm_004](github_tvm_004.py) | TVM Relay | **0.053** | Cubic interpolation diverges from both ORT and PyTorch bicubic |
| 8 | [cross_openvino_conv_bn_fusion](cross_openvino_conv_bn_fusion.py) | OpenVINO 2026.0 | **0.022** | Conv+BatchNorm fusion introduces rounding error above 0.01 tolerance |
| 9 | [github_ort_004](github_ort_004.py) | OnnxRuntime 1.24.4 | — | Optimizer fuses float→int32→bool, skipping truncation step (wrong boolean output) |
| 10 | [github_ort_016](github_ort_016.py) | OnnxRuntime 1.24.4 | — | GridSample bicubic+border clamps after neighbourhood lookup instead of per-sample |
| 11 | [github_ort_008](github_ort_008.py) | OnnxRuntime 1.24.4 | — | CUDA cubic resize antialias grid setup wrong, hardcoded cubic_coeff_a |
| 12 | [github_ort_002](github_ort_002.py) | OnnxRuntime 1.24.4 | — | Nearest resize 1→64 (scale=64/26): pixel selection off-by-one vs PyTorch |
| 13 | [github_onnx_spec_007](github_onnx_spec_007.py) | ONNX Spec | — | Resize nearest half_pixel round_prefer_ceil returns wrong index for element 4 |
| 14 | [cross_openvino_fp16_matmul_add](cross_openvino_fp16_matmul_add.py) | OpenVINO 2026.0 | **0.078** | OpenVINO CPU fp16 MatMul tile-based accumulation diverges from ORT reference; grows with matrix size (64×64 → 0.078, 128×128 → 0.188) |

## By Compiler

| Compiler | Count | Bug IDs |
|---|---|---|
| OnnxRuntime 1.24.4 | 5 | github_ort_002, github_ort_003, github_ort_004, github_ort_008, github_ort_016 |
| onnx2torch / torch.compile | 4 | cross_onnx2torch_cumsum, cross_onnx2torch_resize_nearest_ceil, cross_onnx2torch_resize_nearest_floor, cross_onnx2torch_resize_linear_asym |
| OpenVINO 2026.0 | 2 | cross_openvino_conv_bn_fusion, cross_openvino_fp16_matmul_add |
| TensorFlow XLA | 1 | github_tensorflow_002 |
| TVM Relay | 1 | github_tvm_004 |
| ONNX Spec | 1 | github_onnx_spec_007 |

## New Discoveries (5 bugs)

Bugs #1-3, #5, and #8 are **newly discovered** by the first round of cross-compiler testing.
Bug #14 is **newly discovered** by the extended cross-compiler harness (2026-04-14, 18 new patterns × 5 compilers).

## Cross-Compiler Results (tolerance=0.1)

Full harness tested 23 ONNX patterns × 5 compilers (ORT, OpenVINO, onnx2torch, torch.compile, GPU torch):

| Compiler | Bugs (tol=0.1) | OK | Errors |
|---|---|---|---|
| ORT_opt | 0 | 23 | 0 |
| OpenVINO | 0 | 23 | 0 |
| onnx2torch | 4 | 19 | 0 |
| torch.compile | 4 | 19 | 0 |
| GPU(torch) | 4 | 19 | 0 |

All 4 onnx2torch bugs are Resize and CumSum conversion errors in the onnx2torch library.

## Extended Cross-Compiler Results (tolerance=0.05, 18 new patterns)

Extended harness (`bugs/cross_check_all_bugs.py`) tested 18 additional ONNX patterns × 5 compilers (Groups A–D).
Run date: 2026-04-14. Compilers: ORT 1.24.4, OpenVINO 2026.0, onnx2torch, torch.compile, GPU(torch) 2.9.1+cu128.

| Compiler | Bugs (tol=0.05) | OK | Errors |
|---|---|---|---|
| ORT_opt | 0 | 18 | 0 |
| OpenVINO | 1 | 17 | 0 |
| onnx2torch | 0 | 12 | 6 |
| torch.compile | 0 | 12 | 6 |
| GPU(torch) | 0 | 12 | 6 |

**New bug found**: `cross_openvino_fp16_matmul_add` — OpenVINO CPU fp16 MatMul + Add diverges from ORT reference by 0.078 (64×64), 0.188 (128×128).

Patterns that errored in onnx2torch/torch.compile (not bugs, missing op support):
- `convtranspose_autopad_same` — SAME_UPPER auto_pad not implemented in onnx2torch
- `scatternd_basic` — ScatterND forward() not supported in onnx2torch
- `gridsample_bilinear_zeros` — GridSample converter not in onnx2torch
- `matmul_large_dim` — multi-input model not handled by single-input onnx2torch runner
- `sqrt_div_precision` — multi-input model not handled by single-input onnx2torch runner
- `pixel_shuffle` — DepthToSpace not supported in onnx2torch
