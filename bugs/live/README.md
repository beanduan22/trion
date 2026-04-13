# 12 Live Bugs (Still Reproduce)

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
| 2 | [cross_onnx2torch_resize_nearest_ceil](cross_onnx2torch_resize_nearest_ceil.py) | onnx2torch | **3.00** | onnx2torch uses PyTorch floor nearest_mode instead of ONNX ceil; pixel selection wrong |
| 3 | [github_tensorflow_002](github_tensorflow_002.py) | TensorFlow XLA | **1.00** | MLIR/TOSA nearest resize shifts rows by 1 with half_pixel_centers=True |
| 4 | [cross_onnx2torch_resize_linear_asym](cross_onnx2torch_resize_linear_asym.py) | onnx2torch | **0.80** | ONNX asymmetric coordinate mode mapped to wrong PyTorch interpolation mode |
| 5 | [github_ort_003](github_ort_003.py) | OnnxRuntime 1.24.4 | **0.29** | Asymmetric bilinear resize gives 0.29 max error vs PyTorch reference |
| 6 | [github_tvm_004](github_tvm_004.py) | TVM Relay | **0.053** | Cubic interpolation diverges from both ORT and PyTorch bicubic |
| 7 | [cross_openvino_conv_bn_fusion](cross_openvino_conv_bn_fusion.py) | OpenVINO 2026.0 | **0.022** | Conv+BatchNorm fusion introduces rounding error above 0.01 tolerance |
| 8 | [github_ort_004](github_ort_004.py) | OnnxRuntime 1.24.4 | — | Optimizer fuses float→int32→bool, skipping truncation step (wrong boolean output) |
| 9 | [github_ort_016](github_ort_016.py) | OnnxRuntime 1.24.4 | — | GridSample bicubic+border clamps after neighbourhood lookup instead of per-sample |
| 10 | [github_ort_008](github_ort_008.py) | OnnxRuntime 1.24.4 | — | CUDA cubic resize antialias grid setup wrong, hardcoded cubic_coeff_a |
| 11 | [github_ort_002](github_ort_002.py) | OnnxRuntime 1.24.4 | — | Nearest resize 1→64 (scale=64/26): pixel selection off-by-one vs PyTorch |
| 12 | [github_onnx_spec_007](github_onnx_spec_007.py) | ONNX Spec | — | Resize nearest half_pixel round_prefer_ceil returns wrong index for element 4 |

## By Compiler

| Compiler | Count | Bug IDs |
|---|---|---|
| OnnxRuntime 1.24.4 | 5 | github_ort_002, github_ort_003, github_ort_004, github_ort_008, github_ort_016 |
| onnx2torch / torch.compile | 3 | cross_onnx2torch_cumsum, cross_onnx2torch_resize_nearest_ceil, cross_onnx2torch_resize_linear_asym |
| OpenVINO 2026.0 | 1 | cross_openvino_conv_bn_fusion |
| TensorFlow XLA | 1 | github_tensorflow_002 |
| TVM Relay | 1 | github_tvm_004 |
| ONNX Spec | 1 | github_onnx_spec_007 |

## New Discoveries (4 bugs)

Bugs #1-4 and #7 are **newly discovered** by cross-compiler testing — taking ONNX patterns that were fixed in one compiler and testing them on other compilers.
