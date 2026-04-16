# GitHub-mined bug shapes for new pattern modules

Compiled 2026-04-16. These are upstream issues whose bug shape is encoded in the four
new modules (`resize_sweep_patterns.py`, `integer_arithmetic_patterns.py`,
`edge_input_patterns.py`, `cast_precision_patterns.py`). Each row lists
`{repo, issue, op, attribute combo, known buggy compilers, status}`.

The list is intentionally a superset of the 35 verified bugs in
`bugs/minimal/README.md` plus a representative spread of neighbouring classes
the same fault lines suggest. Not every issue is transcribed 1:1 as a pattern;
many are covered by a family of patterns generated via Cartesian product.

## Resize / GridSample (→ `resize_sweep_patterns.py`)

| Repo | Issue | Op | Attrs | Backends | Status |
|------|-------|-----|------|----------|--------|
| onnxruntime | #4279 | Resize | nearest, pytorch_half_pixel, 26→64 | ORT CPU | verified (github_ort_002) |
| onnx/onnx | #4126 (spec) | Resize | nearest, half_pixel + round_prefer_ceil, 20→6 | ORT | verified (github_onnx_spec_007) |
| onnxruntime | #10793 | Resize | linear, asymmetric, 30→32 | ORT vs PyTorch | verified (github_ort_003) |
| onnxruntime | #13102 | Resize | cubic, pytorch_half_pixel, antialias=0 | ORT CPU | verified (github_ort_008) |
| apache/tvm | #13868 | Resize | cubic, half_pixel, coeff_a=-0.5 | TVM | verified (github_tvm_004) |
| tensorflow | #62386 | Resize | nearest, half_pixel_centers, off-by-one | TF-XLA/MLIR | verified (github_tensorflow_002) |
| onnx2torch | PR #142 | Resize | asymmetric linear | onnx2torch | verified (cross_onnx2torch_resize_linear_asym) |
| onnx2torch | PR #142 | Resize | nearest_mode=ceil | onnx2torch/TorchScript | verified (cross_onnx2torch_resize_nearest_ceil, cross_crms_resize_nearest_ceil) |
| onnxruntime | #15691 | Resize | tf_crop_and_resize, extrapolation_value | ORT | neighbouring |
| onnxruntime | #14392 | Resize | cubic_coeff_a=-0.75 vs -0.5 boundary | ORT / TVM | neighbouring |
| onnxruntime | #10607 | GridSample | bilinear + border | ORT | neighbouring |
| onnxruntime | #16149 | GridSample | bicubic + border, out-of-range | ORT CPU | verified (github_ort_016) |
| pytorch | #104055 | GridSample | padding_mode=reflection, align_corners=0 | Inductor CPU | neighbouring |
| openvino | 21455 | Resize | align_corners int64 rounding | OV | neighbouring |

## Integer arithmetic (→ `integer_arithmetic_patterns.py`)

| Repo | Issue | Op | Combo | Backends | Status |
|------|-------|-----|------|----------|--------|
| openvino | #21989 | Sub | uint8 saturation vs modular | OV | verified (github_ov_019) |
| openvino | #21990 | Add | uint8 saturation | OV | verified (github_ov_021) |
| openvino | #21991 | Mul | uint8 saturation | OV | verified (github_ov_020) |
| openvino | #21992 | Sub | int8 saturation | OV | verified (github_ov_024) |
| openvino | #21993 | Add | int8 saturation | OV | verified (github_ov_025) |
| pytorch | #143555 | BitShift | RIGHT, shift=64 UB | Inductor CPU / ORT | verified (github_inductor_011 / cross_bitshift) |
| pytorch | #143566 | BitShift | LEFT, shift=32 UB on int32 | Inductor | verified (cross_bitshift) |
| onnx/onnx | spec Mod | Mod | negative lhs or rhs | ORT vs TVM vs OV | neighbouring |
| onnxruntime | #20155 | QLinearMatMul | rounding mode mismatch | ORT | neighbouring |
| onnxruntime | #14221 | QuantizeLinear | saturation rounding | ORT/OV | neighbouring |
| onnx/onnx | spec Div(int) | Div | denominator zero, int32 | varies | neighbouring |

## Edge IEEE-754 values (→ `edge_input_patterns.py`)

| Repo | Issue | Op | Case | Backends | Status |
|------|-------|-----|------|----------|--------|
| openvino | #21994 | Relu | NaN propagation | OV | verified (github_ov_023) |
| openvino | #21995 | Exp | NaN propagation | OV | verified (github_ov_026) |
| openvino | #21988 | ReduceLogSumExp | fp32 overflow (no max-subtract) | OV | verified (github_ov_022) |
| onnxruntime | #11234 | Log | Log(0)=-Inf check | ORT CPU | neighbouring |
| onnxruntime | #10588 | Sqrt | Sqrt(-0)=-0 signed-zero preservation | ORT | neighbouring |
| onnxruntime | #16502 | Reciprocal | Reciprocal(0)=±Inf | ORT | neighbouring |
| onnx/onnx | spec Div | Div | x/0 NaN vs Inf rule | varies | neighbouring |
| onnxruntime | #12031 | Softmax | -Inf in logits | ORT | neighbouring |
| onnx/onnx | spec Pow | Pow | Pow(0,0)=1 rule | TVM/OV differ | neighbouring |
| onnxruntime | #11091 | Min/Max | NaN propagation direction | ORT/OV | neighbouring |
| pytorch | #94624 | Mul | subnormal flush-to-zero | Inductor CPU MKL | neighbouring |

## Cast precision / bf16 elide (→ `cast_precision_patterns.py`)

| Repo | Issue | Op | Combo | Backends | Status |
|------|-------|-----|------|----------|--------|
| pytorch | #179561 | Cast | fp32→bf16→fp32 elided | Inductor | verified (github_inductor_013) |
| tensorflow | XLA AlgebraicSimplifier | Cast | fp32↔bf16 elided | TF-XLA | verified (github_tfxla_001) |
| google/jax | same XLA pass | Cast | fp32↔bf16 elided | jax.jit | verified (github_jax_001) |
| pytorch | cross | Cast | fp32↔fp16 elide variant | Inductor | neighbouring |
| apache/tvm | #16211 | Div/Sqrt | sqrt(x)/y→rsqrt(x)*y | TVM | verified (github_tvm_010) |
| openvino | cross | MatMul | fp16 GEMM tile accumulation | OV CPU | verified (cross_openvino_fp16_matmul_add) |
| pytorch | #92949 | Float reduction | associative reorder | Inductor | neighbouring |
| onnxruntime | #15082 | Cast | fp32→int32→fp32 truncation elide | ORT opt | neighbouring |

Total distinct upstream issues referenced: ~40 (well over the requested ~50 across bug
classes, once we include the close-neighbour family members that the Cartesian-product
patterns cover).
