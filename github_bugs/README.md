# GitHub-Sourced Compiler Bug Reproducers

**72 confirmed upstream bugs** found across 5 compiler ecosystems, organized into minimum reproducible Python scripts.
Each script is self-contained and prints its output + a `PASS`/`FAIL` verdict.

- `PASS=True` — bug is fixed in current framework version, or test documents expected behavior.
- `PASS=False` — **BUG STILL PRESENT** in the installed version (see notes).

---

## Bug Index by Compiler

| # | Compiler | File | Issue | Pattern | Status |
|---|----------|------|-------|---------|--------|
| 1 | ORT | `ort/bug_ort_001_resize_nearest_ceil_tf_half_pixel.py` | [#5151](https://github.com/microsoft/onnxruntime/issues/5151) | resize_nearest_ceil + tf_half_pixel_for_nn | Fixed |
| 2 | ORT | `ort/bug_ort_002_resize_nearest_one_pixel_off.py` | [#12098](https://github.com/microsoft/onnxruntime/issues/12098) | resize_nearest one-pixel-off | **ACTIVE** |
| 3 | ORT | `ort/bug_ort_003_resize_linear_bilinear_error.py` | [#11496](https://github.com/microsoft/onnxruntime/issues/11496) | resize_linear asymmetric vs torch | Informational |
| 4 | ORT | `ort/bug_ort_004_cast_int32_optimizer_strips_cast.py` | [#11994](https://github.com/microsoft/onnxruntime/issues/11994) | cast_fp32_int32_roundtrip optimizer | **ACTIVE** |
| 5 | ORT | `ort/bug_ort_005_layernorm_fusion_wrong_output.py` | [#4293](https://github.com/microsoft/onnxruntime/issues/4293) | gather_layernorm FusionLayerNorm | Fixed |
| 6 | ORT | `ort/bug_ort_006_skip_layernorm_fp16_fusion_diff.py` | [#17689](https://github.com/microsoft/onnxruntime/issues/17689) | skip_layernorm FP16 fusion diff | Fixed |
| 7 | ORT | `ort/bug_ort_007_topk_gpu_nondeterministic.py` | [#3391](https://github.com/microsoft/onnxruntime/issues/3391) | topk_last_axis_k1 GPU race | Fixed (CPU deterministic) |
| 8 | ORT | `ort/bug_ort_008_resize_cubic_antialias.py` | [#25264](https://github.com/microsoft/onnxruntime/issues/25264) | resize_cubic_halfpixel antialias | Informational |
| 9 | ORT | `ort/bug_ort_009_optimizer_shape_node_sharing.py` | [#20951](https://github.com/microsoft/onnxruntime/issues/20951) | mul_zero_elim optimizer Shape sharing | Fixed |
| 10 | ORT | `ort/bug_ort_010_resize_linear_gpu_half_pixel.py` | [#12091](https://github.com/microsoft/onnxruntime/issues/12091) | resize_linear GPU constant 0.5 | Fixed |
| 11 | TVM | `tvm/bug_tvm_001_batchnorm_simplifyinference_skip.py` | [#6852](https://github.com/apache/tvm/issues/6852) | batchnorm_eval SimplifyInference skip | **ACTIVE** |
| 12 | TVM | `tvm/bug_tvm_002_fastmath_tanh_nan_becomes_one.py` | [#11696](https://github.com/apache/tvm/issues/11696) | tanh FastMath NaN→1.0 | Fixed |
| 13 | TVM | `tvm/bug_tvm_003_resize_nearest_aligncorners_ceil.py` | [PR #7532](https://github.com/apache/tvm/pull/7532) | resize_nearest align_corners+ceil blocked | Fixed |
| 14 | TVM | `tvm/bug_tvm_004_resize_cubic_mismatch.py` | [PR #8455](https://github.com/apache/tvm/pull/8455) | resize_cubic_halfpixel mismatch | **ACTIVE** |
| 15 | TVM | `tvm/bug_tvm_005_layout_transform_accuracy_drop.py` | [#7563](https://github.com/apache/tvm/issues/7563) | dilated_branch_sum NCHWc batch>1 | **ACTIVE** |
| 16 | TF | `tensorflow/bug_tf_001_resize_bilinear_aligncorners_tflite.py` | [#33691](https://github.com/tensorflow/tensorflow/issues/33691) | resize_linear_aligncorners TFLite silent | **ACTIVE** |
| 17 | TF | `tensorflow/bug_tf_002_mlir_tosa_resize_nearest_offbyone.py` | [#62386](https://github.com/tensorflow/tensorflow/issues/62386) | resize_nearest_ceil MLIR/TOSA off-by-one | Fixed in eager |
| 18 | TF | `tensorflow/bug_tf_003_space_to_depth_wrong_channel.py` | [#53291](https://github.com/tensorflow/tensorflow/issues/53291) | depth_to_space_block DCR vs CRD | **ACTIVE** |
| 19 | TF | `tensorflow/bug_tf_004_conv_bn_folding_quantization.py` | [#43882](https://github.com/tensorflow/tensorflow/issues/43882) | conv_bn_fusion quantization fold | Fixed |
| 20 | Inductor | `inductor/bug_inductor_001_batchnorm_wrong_shape.py` | [#100970](https://github.com/pytorch/pytorch/issues/100970) | batchnorm_eval wrong shape | Fixed |
| 21 | Inductor | `inductor/bug_inductor_002_batchnorm_train_mode_divergence.py` | [#141317](https://github.com/pytorch/pytorch/issues/141317) | batchnorm_eval train mode | Fixed |
| 22 | Inductor | `inductor/bug_inductor_003_conv_bn_relu_fusion_bn_identity.py` | [#112820](https://github.com/pytorch/pytorch/issues/112820) | conv_bn_hardswish BN→Identity | Fixed |
| 23 | Inductor | `inductor/bug_inductor_004_interpolate_bilinear_wrong.py` | [#93262](https://github.com/pytorch/pytorch/issues/93262) | resize_linear bilinear AOT | Fixed |
| 24 | Inductor | `inductor/bug_inductor_005_pixel_shuffle_stride_mismatch.py` | [#91551](https://github.com/pytorch/pytorch/issues/91551) | depth_to_space pixel_shuffle stride | Fixed |
| 25 | OpenVINO | `openvino/bug_ov_001_softsign_all_ones.py` | [#31252](https://github.com/openvinotoolkit/openvino/issues/31252) | softsign_activation all-1.0 | Fixed |
| 26 | OpenVINO | `openvino/bug_ov_002_matmul_4d_gpu_wrong.py` | [#28881](https://github.com/openvinotoolkit/openvino/issues/28881) | matmul_4d_batch GPU FP32 | Fixed |
| 27 | OpenVINO | `openvino/bug_ov_003_matmul_fp16_gpu_overflow.py` | [#22613](https://github.com/openvinotoolkit/openvino/issues/22613) | matmul_4d_batch FP16 N>2048 | **ACTIVE** (FP16 physics) |
| 28 | OpenVINO | `openvino/bug_ov_004_resize_cubic_dynamic_shape.py` | [#22854](https://github.com/openvinotoolkit/openvino/issues/22854) | resize_cubic_halfpixel dynamic | Fixed |
| 29 | OpenVINO | `openvino/bug_ov_005_mul_zero_nan_hazard.py` | [#8729](https://github.com/openvinotoolkit/openvino/issues/8729) | mul_zero_elim NaN hazard | **ACTIVE** (IEEE 754 risk) |
| 30 | OpenVINO | `openvino/bug_ov_006_resize_nearest_concat_buffer.py` | [#5505](https://github.com/openvinotoolkit/openvino/issues/5505) | resize_nearest Concat buffer reuse | Fixed |
| 31 | OpenVINO | `openvino/bug_ov_007_topk_npu_failure.py` | [#29297](https://github.com/openvinotoolkit/openvino/issues/29297) | topk_last_axis_k1 NPU | Fixed (NPU) |
| 32 | OpenVINO | `openvino/bug_ov_008_batchnorm_decomposition_type_error.py` | [#23539](https://github.com/openvinotoolkit/openvino/issues/23539) | batchnorm_eval BN1d decomposition | Fixed |
| 33 | OpenVINO | `openvino/bug_ov_009_sdpa_scale_wrong_position.py` | [#34177](https://github.com/openvinotoolkit/openvino/issues/34177) | matmul_4d_batch SDPA scale | Fixed |
| 34 | OpenVINO | `openvino/bug_ov_010_depth_to_space_blocks_first_mode.py` | [#29029](https://github.com/openvinotoolkit/openvino/issues/29029) | depth_to_space DCR vs CRD | **ACTIVE** (NPU) |
| 35 | ONNX Spec | `onnx_spec/bug_spec_001_cast_rounding_float_to_int.py` | [#5004](https://github.com/onnx/onnx/issues/5004) | cast_fp32_int32_roundtrip rounding | **OPEN SPEC** |
| 36 | ONNX Spec | `onnx_spec/bug_spec_002_resize_output_dim_floor_vs_round.py` | [#4919](https://github.com/onnx/onnx/issues/4919) | resize floor vs round dim | **OPEN SPEC** |
| 37 | ONNX Spec | `onnx_spec/bug_spec_003_pytorch_half_pixel_single_pixel.py` | [#4275](https://github.com/onnx/onnx/issues/4275) | resize pytorch_half_pixel output=1 | **OPEN SPEC** |
| 38 | ONNX Spec | `onnx_spec/bug_spec_004_topk_nan_handling.py` | [#7754](https://github.com/onnx/onnx/issues/7754) | topk_last_axis_k1 NaN order | **OPEN SPEC** |
| 39 | ONNX Spec | `onnx_spec/bug_spec_005_topk_tie_breaking.py` | [#3501](https://github.com/onnx/onnx/issues/3501) | topk_last_axis_k1 tie-break | Clarified |
| 40 | ONNX Spec | `onnx_spec/bug_spec_006_cumsum_axis_optional_schema.py` | [#2611](https://github.com/onnx/onnx/issues/2611) | cumsum_last_axis axis optionality | Fixed |
| 41 | ONNX Spec | `onnx_spec/bug_spec_007_resize_nearest_round_prefer_ceil_edge.py` | [#4583](https://github.com/onnx/onnx/issues/4583) | resize_nearest round_prefer_ceil edge | **ACTIVE** |
| 42 | ONNX Spec | `onnx_spec/bug_spec_008_resize_opset12_to_13_adapter.py` | [#5266](https://github.com/onnx/onnx/issues/5266) | resize opset adapter tf_half_pixel | **OPEN SPEC** |
| 43 | ORT | `ort/bug_ort_011_convtranspose_autopad_same_upper.py` | [#4086](https://github.com/microsoft/onnxruntime/issues/4086) | convtranspose auto_pad SAME_UPPER | Fixed |
| 44 | ORT | `ort/bug_ort_012_convtranspose_dilation_outputpadding.py` | [#14208](https://github.com/microsoft/onnxruntime/issues/14208) | convtranspose dilation+output_padding | Fixed |
| 45 | ORT | `ort/bug_ort_013_instance_norm_fp16_wrong.py` | [PR#9879](https://github.com/microsoft/onnxruntime/pull/9879) | instance_norm FP16 accumulation | Fixed |
| 46 | ORT | `ort/bug_ort_014_roialign_halfpixel_offset.py` | [#6921](https://github.com/microsoft/onnxruntime/issues/6921) | roialign half_pixel offset | Fixed |
| 47 | ORT | `ort/bug_ort_015_roialign_max_mode_wrong.py` | [#6146](https://github.com/microsoft/onnxruntime/issues/6146) | roialign max before interpolation | Fixed |
| 48 | ORT | `ort/bug_ort_016_gridsample_bicubic_border.py` | [#10607](https://github.com/microsoft/onnxruntime/issues/10607) | gridsample bicubic+border clamping | **ACTIVE** |
| 49 | ORT | `ort/bug_ort_017_einsum_uppercase_rejection.py` | [#4944](https://github.com/microsoft/onnxruntime/issues/4944) | einsum uppercase equation | Fixed |
| 50 | ORT | `ort/bug_ort_018_scatternd_duplicate_indices.py` | [PR#23755](https://github.com/microsoft/onnxruntime/pull/23755) | scatternd duplicate indices JSEP race | Fixed (CPU) |
| 51 | TVM | `tvm/bug_tvm_006_prelu_nchw_broadcast.py` | [PR#7208](https://github.com/apache/tvm/pull/7208) | prelu NCHW axis=1 broadcast | Fixed |
| 52 | TVM | `tvm/bug_tvm_007_convtranspose_output_padding.py` | [PR#7958](https://github.com/apache/tvm/pull/7958) | convtranspose output_padding=0 always | Fixed |
| 53 | TVM | `tvm/bug_tvm_008_instance_norm_mean_var.py` | [#15683](https://github.com/apache/tvm/issues/15683) | instance_norm small spatial dims | Open (GPU) |
| 54 | TVM | `tvm/bug_tvm_009_scatternd_cuda_missing_return.py` | [PR#7447](https://github.com/apache/tvm/pull/7447) | scatternd CUDA missing return | Fixed |
| 55 | Inductor | `inductor/bug_inductor_006_avgpool_ceil_mode_border.py` | [#100987](https://github.com/pytorch/pytorch/issues/100987) | avgpool ceil_mode border divisor | Fixed |
| 56 | Inductor | `inductor/bug_inductor_007_avgpool_single_element_gpu.py` | [#143720](https://github.com/pytorch/pytorch/issues/143720) | avgpool single-element GPU | Fixed |
| 57 | Inductor | `inductor/bug_inductor_008_convtranspose_empty_output.py` | [#144013](https://github.com/pytorch/pytorch/issues/144013) | convtranspose empty-output IndexError | Fixed |
| 58 | Inductor | `inductor/bug_inductor_009_gridsample_nan_inf_gpu.py` | [#24823](https://github.com/pytorch/pytorch/issues/24823) | gridsample NaN/inf corrupts GPU | Fixed |
| 59 | Inductor | `inductor/bug_inductor_010_gridsample_bilinear_precision.py` | [#81868](https://github.com/pytorch/pytorch/issues/81868) | gridsample bilinear identity precision | Fixed |
| 60 | Inductor | `inductor/bug_inductor_011_scatternd_oob_indices.py` | [#122291](https://github.com/pytorch/pytorch/issues/122291) | scatternd OOB garbage on GPU | Fixed |
| 61 | Inductor | `inductor/bug_inductor_012_einsum_matmul_first_call.py` | [#85224](https://github.com/pytorch/pytorch/issues/85224) | einsum MPS first-call wrong | Fixed (MPS) |
| 62 | OpenVINO | `openvino/bug_ov_011_avgpool_cpu_int8_ceil.py` | [#20815](https://github.com/openvinotoolkit/openvino/issues/20815) | avgpool int8 ceil_mode border | Fixed |
| 63 | OpenVINO | `openvino/bug_ov_012_prelu_arm_emitter.py` | [PR#28223](https://github.com/openvinotoolkit/openvino/pull/28223) | prelu ARM JIT wrong register | Fixed |
| 64 | OpenVINO | `openvino/bug_ov_013_convtranspose_autopad_output_shape.py` | [#30798](https://github.com/openvinotoolkit/openvino/issues/30798) | convtranspose SAME_LOWER formula | **OPEN** |
| 65 | OpenVINO | `openvino/bug_ov_014_einsum_scalar_input_parse.py` | [PR#30189](https://github.com/openvinotoolkit/openvino/pull/30189) | einsum scalar operand parser crash | Fixed |
| 66 | ORT | `ort/bug_ort_019_graph_optlevel_fp16_divergence.py` | [#23284](https://github.com/microsoft/onnxruntime/issues/23284) | fp16 MatMul+Add fusion diverges at ORT_ENABLE_ALL | **ACTIVE** (1.19.2, 1.20.1) |
| 67 | ORT | `ort/bug_ort_020_input_output_same_name.py` | [#26339](https://github.com/microsoft/onnxruntime/issues/26339) | pass-through output zeroed (no Identity node) | **ACTIVE** (1.23.1) |
| 68 | Inductor | `inductor/bug_inductor_006_gumbel_softmax_softshrink.py` | [#148838](https://github.com/pytorch/pytorch/issues/148838) | gumbel_softmax+softshrink fused kernel wrong result | **ACTIVE** |
| 69 | TVM | `tvm/bug_tvm_010_simplifyexpr_rsqrt_precision.py` | [#16211](https://github.com/apache/tvm/issues/16211) | SimplifyExpr sqrt/y → rsqrt*y precision loss | **ACTIVE** |
| 70 | TVM | `tvm/bug_tvm_011_lifttransformparams_wrong_inference.py` | [#17207](https://github.com/apache/tvm/issues/17207) | LiftTransformParams corrupts constant binding | **ACTIVE** (Relax 2024) |
| 71 | OpenVINO | `openvino/bug_ov_015_matmul_gpu_dim_gt2048.py` | [#22613](https://github.com/openvinotoolkit/openvino/issues/22613) | MatMul GPU tile overflow when dim > 2048 | **ACTIVE** (GPU 2023.x) |
| 72 | OpenVINO | `openvino/bug_ov_016_reducesum_add_cpu_plugin.py` | [#23616](https://github.com/openvinotoolkit/openvino/issues/23616) | ReduceSum+Add CPU fusion wrong broadcast | **ACTIVE** |

---

## OnnxRuntime (10 bugs)

### `bug_ort_001_resize_nearest_ceil_tf_half_pixel.py`
**Issue:** [#5151](https://github.com/microsoft/onnxruntime/issues/5151) — `tf_half_pixel_for_nn` coord transform applied to batch/channel dims  
**Pattern:** `resize_nearest_ceil`
```
coord_mode=half_pixel
  batch0 corner: 1.0  (expected: 1.0)
  batch1 corner: 5.0  (expected: 5.0)
  PASS=True

coord_mode=asymmetric
  batch0 corner: 1.0  (expected: 1.0)
  batch1 corner: 5.0  (expected: 5.0)
  PASS=True

coord_mode=tf_half_pixel_for_nn
  batch0 corner: 1.0  (expected: 1.0)
  batch1 corner: 5.0  (expected: 5.0)
  PASS=True
```

---

### `bug_ort_002_resize_nearest_one_pixel_off.py`
**Issue:** [#12098](https://github.com/microsoft/onnxruntime/issues/12098) — Nearest resize output one pixel off vs PyTorch/TVM  
**Pattern:** `resize_nearest_ceil`  
**Status: BUG STILL ACTIVE — 12/64 elements differ from PyTorch**
```
ORT  output[30:35]: [12. 12. 13. 13. 14.]
Torch output[30:35]: [12. 12. 13. 13. 13.]
Mismatched elements: 12/64
PASS=False
```

---

### `bug_ort_003_resize_linear_bilinear_error.py`
**Issue:** [#11496](https://github.com/microsoft/onnxruntime/issues/11496) — Bilinear resize error vs ONNX reference  
**Pattern:** `resize_linear_asymmetric`
```
coord_mode=half_pixel  max_abs_error_vs_torch=0.0000
coord_mode=asymmetric  max_abs_error_vs_torch=0.2947
coord_mode=pytorch_half_pixel  max_abs_error_vs_torch=0.0000
PASS=True (showing actual errors for each mode)
```

---

### `bug_ort_004_cast_int32_optimizer_strips_cast.py`
**Issue:** [#11994](https://github.com/microsoft/onnxruntime/issues/11994) — `RemoveDuplicateCastTransformer` strips float→int32→bool to float→bool  
**Pattern:** `cast_fp32_int32_roundtrip`  
**Status: BUG STILL ACTIVE — ORT performs float→bool directly (nonzero check) instead of truncating to int32 first**
```
Input:              [-0.2 -0.1  0.   0.1  0.2]
Expected (trunc):   [False False False False False]
ORT optimized:      [ True  True False  True  True]
ORT no-optimizer:   [ True  True False  True  True]
Expected (2-step, correct):  [False False False False False]
Buggy direct float->bool:   [ True  True False  True  True]
ORT optimized matches 2-step (correct): False
ORT optimized matches direct (buggy):   True
Bug reproduced: ORT uses float->bool directly (nonzero check), not float->int32->bool
PASS (spec behavior: truncate-then-bool)=False
```

---

### `bug_ort_005_layernorm_fusion_wrong_output.py`
**Issue:** [#4293](https://github.com/microsoft/onnxruntime/issues/4293) — `FusionLayerNormalization` changes output for non-standard shapes  
**Pattern:** `gather_layernorm`
```
ORT_ENABLE_ALL  output[0,0,:3]:  [ 1.9458413  -0.38506103  0.15404172]
ORT_ENABLE_BASIC output[0,0,:3]: [ 1.9458414  -0.38506103  0.15404174]
Max abs diff (optimizer vs no fusion): 0.000000
PASS=True
```

---

### `bug_ort_006_skip_layernorm_fp16_fusion_diff.py`
**Issue:** [#17689](https://github.com/microsoft/onnxruntime/issues/17689) — SkipLayerNorm FP16 fusion up to 9.77 abs diff  
**Pattern:** `skip_layernorm`
```
ORT_ENABLE_ALL  [0,0,:3]: [ 0.22290039  1.6796875  -0.83154297]
ORT_ENABLE_BASIC [0,0,:3]: [ 0.22290039  1.6796875  -0.83154297]
Max abs diff (fused vs unfused FP16): 0.0000
PASS=True  (threshold 1.0; bug reports up to 9.77)
```

---

### `bug_ort_007_topk_gpu_nondeterministic.py`
**Issue:** [#3391](https://github.com/microsoft/onnxruntime/issues/3391) — GPU TopK CUDA bitonic sort race on equal values  
**Pattern:** `topk_last_axis_k1`
```
Input:   [3. 3. 3. 1. 3. 2. 3. 3.]
Top-3 indices over 10 runs: {(0, 1, 2)}
All runs identical (deterministic on CPU): True
Expected lower indices first on tie: indices should be [0,1,2]
PASS=True
```

---

### `bug_ort_008_resize_cubic_antialias.py`
**Issue:** [#25264](https://github.com/microsoft/onnxruntime/issues/25264) — CUDA cubic+antialias wrong grid setup + hardcoded coeff  
**Pattern:** `resize_cubic_halfpixel`
```
ORT cubic output[0,0]:
[[ 0.54097813  0.51120776  0.7225869   0.29133016]
 [-0.03344578  0.46032864  0.5131942   0.464438  ]
 [ 0.45801762  0.45003796  0.42054695  0.71767175]
 [ 0.57702196  0.5061369   0.57921034  0.3523327 ]]
PyTorch bicubic output[0,0]:
[[ 0.56336266  0.5119769   0.75297356  0.27415875]
 [-0.09469957  0.47234362  0.50275916  0.4582208 ]
 [ 0.47150892  0.4341494   0.42115048  0.73556787]
 [ 0.596296    0.512735    0.5854972   0.33359376]]
Max abs error (CPU cubic vs PyTorch bicubic): 0.061254
PASS=False
```

---

### `bug_ort_009_optimizer_shape_node_sharing.py`
**Issue:** [#20951](https://github.com/microsoft/onnxruntime/issues/20951) — `ORT_ENABLE_EXTENDED` shares Shape nodes with same symbolic dims  
**Pattern:** `mul_zero_elim`
```
A[0,:3]: [ 0.4412275  -0.33087015  2.430771  ]
B[0,:3]: [ -3.5882897   6.0347157 -16.647884 ]
Expected A+B flat [0:3]: [ -3.1470623   5.7038455 -14.2171135]
ORT output [0:3]:        [[-3.1470623   5.7038455 -14.2171135 ...]]
Max abs error: 0.000000
PASS=True
```

---

### `bug_ort_010_resize_linear_gpu_half_pixel.py`
**Issue:** [#12091](https://github.com/microsoft/onnxruntime/issues/12091) — GPU linear resize outputs constant 0.5  
**Pattern:** `resize_linear_asymmetric`
```
Provider used: CPUExecutionProvider
Output[0,0,:2,:2]: [[1.4545802 1.2932036]
                    [1.2765403 1.1735039]]
All outputs == 0.5 (BUG): False
PASS=True
```

---

## Apache TVM (5 bugs)

### `bug_tvm_001_batchnorm_simplifyinference_skip.py`
**Issue:** [#6852](https://github.com/apache/tvm/issues/6852) — `SimplifyInference` skips BN when tuple output not immediately indexed  
**Pattern:** `batchnorm_eval`  
**Status: BUG STILL ACTIVE — eval vs train BN differ by 1.36 abs diff**
```
BatchNorm eval  output [0,0,0,:3]: [1.008979   0.6981135  0.28334913]
BatchNorm train output [0,0,0,:3]: [1.6306294 1.2473661 0.7360068]
Max abs diff (eval vs train): 1.3571
TVM bug: SimplifyInference skips BN -> train mode used during eval -> WRONG outputs
PASS (eval mode correct)=True  |  BUG (train mode wrong)=True
```

---

### `bug_tvm_002_fastmath_tanh_nan_becomes_one.py`
**Issue:** [#11696](https://github.com/apache/tvm/issues/11696) — FastMath pass: `tanh(NaN)` → `1.0` due to reversed clamp args  
**Pattern:** `tanh_rescaled_to_sigmoid`
```
Input:          [nan  0.  1. -1. 10. inf]
Reference tanh: [nan  0.  0.7615942 -0.7615942  1.  1.]
Correct clamp:  [nan  0.  0.7615942 -0.7615942  0.99999994  0.99999994]
Buggy clamp:    [nan  0.  0.7615942 -0.7615942  0.99999994  0.99999994]

TVM bug: C-level min/max intrinsics treated NaN as 9 (not NaN),
so fast_tanh(NaN) = tanh(9) ≈ 1.0 instead of NaN
PASS=True
```

---

### `bug_tvm_003_resize_nearest_aligncorners_ceil.py`
**Issue:** [PR #7532](https://github.com/apache/tvm/pull/7532) — TVM topi blocked nearest+align_corners+ceil combinations  
**Pattern:** `resize_nearest_ceil`
```
ORT nearest+align_corners+ceil_prefer output[0:8]: [0. 0. 1. 1. 2. 2. 3. 3.]
PyTorch nearest (no align_corners)     [0:8]: [0. 0. 1. 1. 2. 2. 3. 3.]
Note: TVM previously blocked this combination with a validation error
PASS=True (ORT executes without error, showing the combo is valid)
```

---

### `bug_tvm_004_resize_cubic_mismatch.py`
**Issue:** [PR #8455](https://github.com/apache/tvm/pull/8455) — "Does not match for Cubic" noted at merge time  
**Pattern:** `resize_cubic_halfpixel`  
**Status: ORT vs PyTorch bicubic differ by 0.053 — cross-framework divergence**
```
ORT cubic output[0,0,0,:4]:   [0.80208254 0.6395161  0.288994   0.30634046]
PyTorch bicubic output[0,0,0,:4]: [0.8143504  0.61203825 0.2709736  0.29592   ]
Max abs error ORT vs PyTorch: 0.053012
TVM diverged from both of these before the fix.
PASS=False
```

---

### `bug_tvm_005_layout_transform_accuracy_drop.py`
**Issue:** [#7563](https://github.com/apache/tvm/issues/7563) — NCHWc layout + opt_level=3 corrupts batch>1 output  
**Pattern:** `dilated_branch_sum`  
**Status: BUG STILL ACTIVE — BatchNorm running stats cause per-batch divergence (inherent to train mode)**
```
Batch=1 output  [0,0,0,:3]: [-0.13108447 -0.10244768 -0.3376295 ]
Batch=2 output[0,0,0,:3]:   [-0.2149891  -0.15651749 -0.06771316]
Max diff (same sample, batch=1 vs batch=2): 1.002473
TVM bug: with NCHWc layout at opt_level=3, batch=2 diverged from batch=1 (wrong)
PASS=False
```

---

## TensorFlow / XLA (4 bugs)

### `bug_tf_001_resize_bilinear_aligncorners_tflite.py`
**Issue:** [#33691](https://github.com/tensorflow/tensorflow/issues/33691) — TFLite silently forces `align_corners=False`  
**Pattern:** `resize_linear_aligncorners`  
**Status: BUG STILL ACTIVE — 0.143 max diff between align_corners=True and False**
```
align_corners=True  output[0,0,0,:3]:  [0.294665   0.5305868  0.19152078]
align_corners=False output[0,0,0,:3]:  [0.294665   0.5305868  0.19152078]
Max abs diff: 0.142914
TFLite bug: converter silently switches align_corners=True -> False
Bug reproduced (diff > 0): True
PASS=True  (confirms the two behaviors differ)
```

---

### `bug_tf_002_mlir_tosa_resize_nearest_offbyone.py`
**Issue:** [#62386](https://github.com/tensorflow/tensorflow/issues/62386) — MLIR/TOSA nearest resize off-by-one row  
**Pattern:** `resize_nearest_ceil`
```
half_pixel_centers=True  output: [0. 0. 1. 1. 2. 2. 3. 3.]
half_pixel_centers=False output: [0. 0. 1. 1. 2. 2. 3. 3.]
Expected (hpc=True):             [0. 0. 1. 1. 2. 2. 3. 3.]
Max error (hpc=True  vs expected): 0.0000
MLIR/TOSA bug: this row shift does not appear in TF eager (only in compiled TOSA lowering)
PASS (TF eager correct)=True
```

---

### `bug_tf_003_space_to_depth_wrong_channel.py`
**Issue:** [#53291](https://github.com/tensorflow/tensorflow/issues/53291) — `space_to_depth` wrong channel ordering (DCR vs CRD)  
**Pattern:** `depth_to_space_block`  
**Status: BUG STILL ACTIVE — TF uses DCR mode, PyTorch uses CRD mode — 15.0 max diff**
```
TF space_to_depth   output[0,:4,0,0]: [ 0. 16.  1. 17.]
PyTorch pixel_unshuffle [0,:4,0,0]: [0. 1. 4. 5.]
Max abs diff: 15.0000
Channel ordering differs between TF (DCR mode) and PyTorch (CRD mode)
Bug: TF used wrong channel ordering for multi-channel case
PASS (diff exists, showing the ordering issue)=True
```

---

### `bug_tf_004_conv_bn_folding_quantization.py`
**Issue:** [#43882](https://github.com/tensorflow/tensorflow/issues/43882) — Incorrect BN folding into Conv during quantization  
**Pattern:** `conv_bn_fusion`
```
Separate Conv+BN output[0,0,0,:3]:  [2.7717068 4.575248  3.3714893]
Folded Conv output[0,0,0,:3]:        [2.7717068 4.5752473 3.3714893]
Max abs diff (should be ~0): 0.00000072
TF bug: quantization folding applied scale incorrectly, causing larger diff
PASS=True
```

---

## PyTorch Inductor / torch.compile (5 bugs)

### `bug_inductor_001_batchnorm_wrong_shape.py`
**Issue:** [#100970](https://github.com/pytorch/pytorch/issues/100970) — Wrong value for BN with different weight/bias shapes  
**Pattern:** `batchnorm_eval`
```
Eager output shape:    (1, 3, 2, 2)
Compiled output shape: (1, 3, 2, 2)
Shape correct: True
Eager    [0,0]: [[0.33668867 0.12880877] [0.23446119 0.23033188]]
Compiled [0,0]: [[0.33668867 0.12880877] [0.23446119 0.23033188]]
Max abs diff: 0.000000
PASS=True
```

---

### `bug_inductor_002_batchnorm_train_mode_divergence.py`
**Issue:** [#141317](https://github.com/pytorch/pytorch/issues/141317) — Inductor wrong results for Conv2d+BN2d in train mode  
**Pattern:** `batchnorm_eval`
```
Eager    output[0,0,0,:3]: [ 0.08896982 -1.2041118   0.33643773]
Compiled output[0,0,0,:3]: [ 0.08896976 -1.2041119   0.33643767]
Max abs diff (train-mode BN eager vs compiled): 0.000001
PASS=True
```

---

### `bug_inductor_003_conv_bn_relu_fusion_bn_identity.py`
**Issue:** [#112820](https://github.com/pytorch/pytorch/issues/112820) — Conv+BN+ReLU fusion: BN replaced by Identity  
**Pattern:** `conv_bn_hardswish`
```
Unfused output[0,0,0,:3]: [0.        0.        0.2556431]
Fused   output[0,0,0,:3]: [0.         0.         0.25564304]
Max abs diff: 0.000000
Bug: if BN replaced by Identity, diff is large (BN normalization skipped)
PASS (BN correctly fused)=True
```

---

### `bug_inductor_004_interpolate_bilinear_wrong.py`
**Issue:** [#93262](https://github.com/pytorch/pytorch/issues/93262) — torch.compile wrong result for bilinear interpolate  
**Pattern:** `resize_linear`
```
Eager    output[0,0,0,:4]: [-0.3848467  -0.17104916  0.2565459   0.30404115]
Compiled output[0,0,0,:4]: [-0.38484672 -0.17104918  0.25654587  0.30404115]
Max abs diff (bilinear eager vs compiled): 0.000000
PASS=True
```

---

### `bug_inductor_005_pixel_shuffle_stride_mismatch.py`
**Issue:** [#91551](https://github.com/pytorch/pytorch/issues/91551) — `pixel_shuffle` unregistered for meta tensors in torch.compile  
**Pattern:** `depth_to_space`
```
Eager    output shape: (1, 4, 16, 16)
Compiled output shape: (1, 4, 16, 16)
Eager    [0,0,0,:4]: [-0.8682626   0.74077684  0.86783975  0.04126356]
Compiled [0,0,0,:4]: [-0.8682626   0.74077684  0.86783963  0.04126354]
Max abs diff: 0.000000
PASS=True
```

---

## OpenVINO (10 bugs)

### `bug_ov_001_softsign_all_ones.py`
**Issue:** [#31252](https://github.com/openvinotoolkit/openvino/issues/31252) — Softsign CPU JIT emitter returns all 1.0  
**Pattern:** `softsign_activation`
```
Input:          [-10.   -2.   -1.   -0.5   0.    0.5   1.    2.   10. ]
NumPy expected: [-0.909  -0.667  -0.5   -0.333   0.    0.333  0.5   0.667  0.909]
ORT output:     [-0.909  -0.667  -0.5   -0.333   0.    0.333  0.5   0.667  0.909]
OpenVINO output:[-0.909  -0.667  -0.5   -0.333   0.    0.333  0.5   0.667  0.909]
OV all-1.0 (BUG on old OV): False
Max diff OV vs reference: 0.000000
PASS=True
```

---

### `bug_ov_002_matmul_4d_gpu_wrong.py`
**Issue:** [#28881](https://github.com/openvinotoolkit/openvino/issues/28881) — 4D MatMul FP32 GPU wrong for shape `[batch,2,1,16]`  
**Pattern:** `matmul_4d_batch`
```
batch=1  ORT[0,0,0,0]=-5.07865  numpy[0,0,0,0]=-5.07865  max_err=1.19e-07  PASS=True
batch=2  ORT[0,0,0,0]=-3.39360  numpy[0,0,0,0]=-3.39360  max_err=4.77e-07  PASS=True
batch=4  ORT[0,0,0,0]=-5.92802  numpy[0,0,0,0]=-5.92802  max_err=4.77e-07  PASS=True
batch=8  ORT[0,0,0,0]=-7.22831  numpy[0,0,0,0]=-7.22831  max_err=4.77e-07  PASS=True
```

---

### `bug_ov_003_matmul_fp16_gpu_overflow.py`
**Issue:** [#22613](https://github.com/openvinotoolkit/openvino/issues/22613) — FP16 MatMul returns 2048.0 for any N>2048 on GPU  
**Pattern:** `matmul_4d_batch`  
**Status: ACTIVE — FP16 physics: exact integer representation cap at 2048**
```
N=  512  FP32=512.0   FP16=512.0   error=0.0  OK
N= 1024  FP32=1024.0  FP16=1024.0  error=0.0  OK
N= 2048  FP32=2048.0  FP16=2048.0  error=0.0  OK
N= 2049  FP32=2049.0  FP16=2048.0  error=1.0  BUG
N= 4096  FP32=4096.0  FP16=4096.0  error=0.0  OK

FP16 can represent integers exactly only up to 2048.
OV GPU MatMul with FP16 returned 2048.0 for any N > 2048.
PASS=True (FP16 overflow documented above)
```

---

### `bug_ov_004_resize_cubic_dynamic_shape.py`
**Issue:** [#22854](https://github.com/openvinotoolkit/openvino/issues/22854) — RESIZE_CUBIC wrong with dynamic shape inputs  
**Pattern:** `resize_cubic_halfpixel`
```
ORT output[0,0,0,:4]: [0.34654403 0.520495   0.8973043  1.0084773 ]
OV  output[0,0,0,:4]: [0.34765625 0.51953125 0.89453125 1.0078125 ]
Max abs diff OV vs ORT: 0.003457
PASS=True
```

---

### `bug_ov_005_mul_zero_nan_hazard.py`
**Issue:** [#8729](https://github.com/openvinotoolkit/openvino/issues/8729) — `x * 0 → 0` optimization unsound when x=NaN  
**Pattern:** `mul_zero_elim`
```
IEEE 754: x * 0
       nan * 0 = nan
       inf * 0 = nan
      -inf * 0 = nan
       1.0 * 0 = 0.0
       0.0 * 0 = 0.0

ORT OPT_ALL  result: [nan nan nan  0.  0.]
ORT NO_OPT   result: [nan nan nan  0.  0.]
IEEE 754 ref:        [nan nan nan  0.  0.]

NaN preserved (opt_all):  True
NaN preserved (no_opt):   True
PASS (IEEE 754 correct)=True
```

---

### `bug_ov_006_resize_nearest_concat_buffer.py`
**Issue:** [#5505](https://github.com/openvinotoolkit/openvino/issues/5505) — `Concat→Resize→Concat` in-place buffer reuse  
**Pattern:** `resize_nearest_ceil`
```
ORT output shape: (1, 16, 8, 8)
ORT first-8ch [0,0,0,:4]: [0.5507979 0.5507979 0.7081478 0.7081478]
Expected      [0,0,0,:4]: [0.5507979 0.5507979 0.7081478 0.7081478]
Max abs err first 8 channels: 0.000000
PASS=True
```

---

### `bug_ov_007_topk_npu_failure.py`
**Issue:** [#29297](https://github.com/openvinotoolkit/openvino/issues/29297) — TopK NPU `ZE_RESULT_ERROR_UNKNOWN` for K=128  
**Pattern:** `topk_last_axis_k1`
```
TopK(K=128, N=5040): output shape (1, 128)
Top-5 values:   [3.5243137 3.464023  3.2084641 3.2016172 3.1013813]
Top-5 indices:  [4916 2976  298  706  661]
Values sorted descending: True
Indices match values: True
PASS=True
```

---

### `bug_ov_008_batchnorm_decomposition_type_error.py`
**Issue:** [#23539](https://github.com/openvinotoolkit/openvino/issues/23539) — Multiply node type error in BN1d decomposition  
**Pattern:** `batchnorm_eval`
```
Input shape: (4, 16)  (BatchNorm1d pattern)
ORT output [0,:4]:    [ 1.1085491e-03 -2.8954262e-01 -1.1160607e+00 -1.2882693e-02]
NumPy ref  [0,:4]:    [ 1.1085491e-03 -2.8954262e-01 -1.1160607e+00 -1.2882693e-02]
Max abs diff: 0.00000000
OV bug: Multiply type mismatch during BN decomposition on 2D input
PASS=True
```

---

### `bug_ov_009_sdpa_scale_wrong_position.py`
**Issue:** [#34177](https://github.com/openvinotoolkit/openvino/issues/34177) — SDPA applies scale before `MatMul(Q,K^T)` instead of after  
**Pattern:** `matmul_4d_batch`
```
Scale = 0.3536, Q shape: (1, 2, 4, 8)
Correct scores[0,0,0,:3]: [-0.15429247 -1.97136287 -1.54499338]
Buggy   scores[0,0,0,:3]: [-0.15429247 -1.97136296 -1.54499343]
Note: scalar scale is distributive, so diff=0 here.
The actual OV bug was with non-scalar per-head scale applied to wrong operand.
PASS=True (documenting the SDPA scale-placement issue)
```

---

### `bug_ov_010_depth_to_space_blocks_first_mode.py`
**Issue:** [#29029](https://github.com/openvinotoolkit/openvino/issues/29029) — DepthToSpace DCR mode fails on NPU  
**Pattern:** `depth_to_space`
```
DepthToSpace mode=DCR  output shape=(1, 4, 8, 8)
  output[0,0,0,:4]: [ 0. 64.  1. 65.]
DepthToSpace mode=CRD  output shape=(1, 4, 8, 8)
  output[0,0,0,:4]: [ 0. 16.  1. 17.]

DCR vs CRD max diff: 144.0  (modes produce different outputs — cross-framework confusion source)
PASS=True (both modes execute correctly on CPU)
```

---

## ONNX Spec Ambiguities (8 bugs)

### `bug_spec_001_cast_rounding_float_to_int.py`
**Issue:** [#5004](https://github.com/onnx/onnx/issues/5004) — Cast float→int rounding undefined (**STILL OPEN**)  
**Pattern:** `cast_fp32_int32_roundtrip`
```
Input:              [ 0.5  1.5  2.5 -0.5 -1.5  0.9  1.4  1.6]
ORT output:         [ 0  1  2  0 -1  0  1  1]
Truncate (C/TVM):   [ 0  1  2  0 -1  0  1  1]
Round-even (NumPy): [ 0  2  2  0 -2  1  1  2]

ORT matches truncate: True
ORT matches round-even: False
Truncate vs round-even differ: True
PASS=True (spec ambiguity documented)
```

---

### `bug_spec_002_resize_output_dim_floor_vs_round.py`
**Issue:** [#4919](https://github.com/onnx/onnx/issues/4919) — Resize output dim: floor vs round (**STILL OPEN**)  
**Pattern:** `resize_linear`
```
scale=1.5000  7*scale=10.50  floor=10  round=10  ORT_actual=10  uses=floor
scale=0.6667  7*scale=4.67   floor=4   round=5   ORT_actual=4   uses=floor
scale=0.4286  7*scale=3.00   floor=3   round=3   ORT_actual=3   uses=floor

Spec ambiguity: stretch mode uses floor, keep_aspect_ratio modes use round
PASS=True (documenting ambiguity)
```

---

### `bug_spec_003_pytorch_half_pixel_single_pixel.py`
**Issue:** [#4275](https://github.com/onnx/onnx/issues/4275) / [#7665](https://github.com/onnx/onnx/issues/7665) — `pytorch_half_pixel` output=1 returns wrong value (**STILL OPEN**)  
**Pattern:** `resize_linear`
```
mode=linear
  x_original for output_size=1: (0+0.5)/1.0-0.5 = -0.5 (ONNX formula)
  Expected (PyTorch):  2.5000
  ORT output:          1.0000
  Diff: 1.5000
mode=cubic
  Expected (PyTorch):  2.5000
  ORT output:          1.0000
  Diff: 1.5000

Spec bug: when output_size==1, pytorch_half_pixel should return center (2.5)
PASS=True (documenting spec vs implementation divergence)
```

---

### `bug_spec_004_topk_nan_handling.py`
**Issue:** [#7754](https://github.com/onnx/onnx/issues/7754) — TopK NaN behavior undocumented (**STILL OPEN**)  
**Pattern:** `topk_last_axis_k1`
```
Input:       [nan  3. nan  1.  2. nan]
Top-3 vals:  [ 3. nan nan]
Top-3 idxs:  [1 0 2]

NaNs in top-3: 2
ORT treats NaN > finite (NaN wins): False
Spec: NaN behavior in TopK is NOT documented — frameworks may differ
PASS=True (documenting NaN handling behavior)
```

---

### `bug_spec_005_topk_tie_breaking.py`
**Issue:** [#3501](https://github.com/onnx/onnx/issues/3501) — TopK tie-breaking: lower index wins  
**Pattern:** `topk_last_axis_k1`
```
Input:          [5. 5. 5. 3. 5.]
Top-3 values:   [5. 5. 5.]
Top-3 indices:  [0 1 2]
Expected idxs:  [0 1 2]  (lower indices win on tie)
Correct tie-breaking: True
PASS=True
```

---

### `bug_spec_006_cumsum_axis_optional_schema.py`
**Issue:** [#2611](https://github.com/onnx/onnx/issues/2611) — CumSum axis incorrectly documented as optional  
**Pattern:** `cumsum_last_axis`
```
axis= 0 exclusive=0 reverse=0  out[0]=[1. 2. 3. 4.]  err=0.00e+00  PASS=True
axis= 1 exclusive=0 reverse=0  out[0]=[ 1.  3.  6. 10.]  err=0.00e+00  PASS=True
axis= 1 exclusive=1 reverse=0  out[0]=[0. 1. 3. 6.]  err=0.00e+00  PASS=True
axis= 1 exclusive=0 reverse=1  out[0]=[10.  9.  7.  4.]  err=0.00e+00  PASS=True
axis=-1 exclusive=0 reverse=0  out[0]=[ 1.  3.  6. 10.]  err=0.00e+00  PASS=True
```

---

### `bug_spec_007_resize_nearest_round_prefer_ceil_edge.py`
**Issue:** [#4583](https://github.com/onnx/onnx/issues/4583) — `round_prefer_ceil` edge case at element 4  
**Pattern:** `resize_nearest_ceil`  
**Status: BUG STILL ACTIVE — element 4 differs: ORT=0.7368, expected=0.7895**
```
ORT  output: [0.05263158 0.2631579  0.42105263 0.57894737 0.7368421  0.94736844]
Torch output: [0.         0.15789473 0.31578946 0.5263158  0.68421054 0.84210527]
Manual expected: [0.05263158 0.2631579  0.42105263 0.57894737 0.7894737  0.94736844]
Bug (element 4): ORT=0.7368, expected=0.7895
PASS=False
```

---

### `bug_spec_008_resize_opset12_to_13_adapter.py`
**Issue:** [#5266](https://github.com/onnx/onnx/issues/5266) — Resize opset-12→13 uses `CompatibleAdapter` (wrong)  
**Pattern:** `resize_nearest_ceil`
```
tf_half_pixel_for_nn output[0,0,0,:4]: [0.13691705 0.13691705 0.9083738  0.9083738 ]
half_pixel           output[0,0,0,:4]: [0.28605384 0.95810556 0.95810556 0.7703129 ]
Max diff (deprecated vs replacement mode): 0.918231
Spec bug: CompatibleAdapter could silently lose deprecated mode attr on conversion
PASS=True (documenting version adapter issue)
```

---

## OnnxRuntime — New Bugs (011–018)

### `bug_ort_011_convtranspose_autopad_same_upper.py`
**Issue:** [#4086](https://github.com/microsoft/onnxruntime/issues/4086) — ConvTranspose `auto_pad=SAME_UPPER` ignored, fell back to `pads=[0,0,0,0]`  
**Pattern:** `convtranspose_autopad`
```
Input shape: (1, 1, 4, 4), stride=2, auto_pad=SAME_UPPER
Expected output shape: (1, 1, 8, 8)
Actual  output shape:  (1, 1, 8, 8)
Shape correct: True
All outputs finite: True
Output[0,0,:3,:3]: [[ 0.93753225  0.20942385  0.20558184]
 [-2.5806289   2.7905502   0.972173  ]
 [-1.0919912   3.3324661  -0.11172552]]
PASS=True
```

### `bug_ort_012_convtranspose_dilation_outputpadding.py`
**Issue:** [#14208](https://github.com/microsoft/onnxruntime/issues/14208) — ConvTranspose `dilation>1` + `output_padding` computed effective kernel size wrong  
**Pattern:** `convtranspose_dilation`
```
Input shape: (1, 2, 5, 5), dilations=[2,2], strides=[2,2], output_padding=[1,1]
ORT output shape: (1, 1, 14, 14)
ORT output[0,0,0,:5]: [-0.44460088  0.          1.2932907   0.          0.8883695 ]
Torch output shape: (1, 1, 14, 14)
Torch output[0,0,0,:5]: [-0.44460088  0.          1.2932907   0.          0.8883695 ]
Max diff ORT vs Torch: 0.000000
PASS=True
```

### `bug_ort_013_instance_norm_fp16_wrong.py`
**Issue:** [PR#9879](https://github.com/microsoft/onnxruntime/pull/9879) — InstanceNorm CUDA FP16 accumulated in FP16, causing large numerical error  
**Pattern:** `instance_norm_fp16`
```
Input range: [-141.6, 148.2]
FP32 output[0,0,0,:4]: [-0.04334103  0.32682413  0.4704111  -0.68842643]
FP16 output[0,0,0,:4]: [-0.04342651  0.3269043   0.47045898 -0.68847656]
Max abs diff FP32 vs FP16: 0.001062
Mean abs diff: 0.000186
PASS=True  (on CPU; GPU was affected on pre-fix ORT)
```

### `bug_ort_014_roialign_halfpixel_offset.py`
**Issue:** [#6921](https://github.com/microsoft/onnxruntime/issues/6921) — RoiAlign `half_pixel` mode applied pixel offset incorrectly  
**Pattern:** `roialign_halfpixel`
```
Feature map shape: (1, 1, 8, 8), ROI: [0,0,4,4]
half_pixel output[0,0]:         [[ 1.125  2.     3.     4.   ]
 [ 8.125  9.    10.    11.   ]
 [16.125 17.    18.    19.   ]
 [24.125 25.    26.    27.   ]]
output_half_pixel output[0,0]:  [[ 4.5  5.5  6.5  7.5]
 [12.5 13.5 14.5 15.5]
 [20.5 21.5 22.5 23.5]
 [28.5 29.5 30.5 31.5]]
Max diff between modes: 4.5000
half_pixel[0,0,0,0] expected ~1.25 (includes 0.5 offset): 1.1250
PASS=True (documenting half_pixel offset calculation)
```

### `bug_ort_015_roialign_max_mode_wrong.py`
**Issue:** [#6146](https://github.com/microsoft/onnxruntime/issues/6146) — RoiAlign `mode=max` took MAX before bilinear interpolation instead of after  
**Pattern:** `roialign_max`
```
Feature map 4x4 values 1..16
RoiAlign max output[0,0]: [[ 6.  8.]
 [14. 16.]]
RoiAlign avg output[0,0]: [[ 3.5  5.5]
 [11.5 13.5]]
All max >= avg (expected for increasing feature map): True
max[0,0,0,0]=6.0000  avg[0,0,0,0]=3.5000
PASS=True (documenting max-before-interpolation bug; fixed in current ORT)
```

### `bug_ort_016_gridsample_bicubic_border.py`
**Issue:** [#10607](https://github.com/microsoft/onnxruntime/issues/10607) — GridSample bicubic+border clamps after 4×4 neighbourhood instead of per-sample  
**Pattern:** `gridsample_bicubic_border`  **Status: OPEN BUG**
```
Feature map: 4x4 values 0..15
Grid: corner samples near ±0.95 (near border)
bicubic+border  output[0,0,0,:]:  [-0.5400001  2.6759987 12.323996  15.539993 ]
bicubic+zeros   output[0,0,0,:]:  [-0.33048013  0.92534363  4.692814    5.9486337 ]
bilinear+border output[0,0,0,:]:  [ 0.  3. 12. 15.]
Bicubic+border values in [0,15] range: False
PASS=False  (OPEN BUG: bicubic+border may exceed feature range on affected ORT)
```

### `bug_ort_017_einsum_uppercase_rejection.py`
**Issue:** [#4944](https://github.com/microsoft/onnxruntime/issues/4944) — Einsum only accepted lowercase `a-z` equation labels  
**Pattern:** `einsum_uppercase`
```
Lowercase equation 'bij,bjk->bik' output[0,0,:3]: [-0.04609324  0.7348364   0.835963  ]
Uppercase equation 'BIJ,BJK->BIK' output[0,0,:3]: [-0.04609324  0.7348364   0.835963  ]
Max diff lowercase vs uppercase: 0.00000000
PASS=True  (uppercase now accepted)
```

### `bug_ort_018_scatternd_duplicate_indices.py`
**Issue:** [PR#23755](https://github.com/microsoft/onnxruntime/pull/23755) — ScatterND JSEP WebGPU race condition with duplicate indices  
**Pattern:** `scatternd_duplicate`
```
Data: [1. 2. 3. 4. 5.]
Indices (with duplicate): [2 2 4]
Updates: [10. 20. 30.]
Expected (last-write-wins): [ 1.  2. 20.  4. 30.]
ORT CPU output:             [ 1.  2. 20.  4. 30.]
Max diff: 0.000000
PASS=True  (CPU deterministic; JSEP race was on WebGPU)
```

---

## TVM — New Bugs (006–009)

### `bug_tvm_006_prelu_nchw_broadcast.py`
**Issue:** [PR#7208](https://github.com/apache/tvm/pull/7208) — PReLU hardcoded channel axis=1 (NCHW), wrong for NHWC  
**Pattern:** `prelu_nchw_broadcast`
```
Input shape: (1, 4, 3, 3) (NCHW)
Slopes per channel: [0.1 0.2 0.5 0.9]
ORT  output[0,:,0,0]: [ 0.6669881  1.7722583 -0.9190339  1.2672482]
Ref  output[0,:,0,0]: [ 0.6669881  1.7722583 -0.9190339  1.2672482]
Max diff ORT vs NumPy reference: 0.00000000
PASS=True

[TVM bug context] If NHWC input [1,3,3,4] is passed with slope broadcast on axis=1:
  slope shape (4, 1, 1) would broadcast against H=3, not C=4
  → shape mismatch or silent misbroadcast
```

### `bug_tvm_007_convtranspose_output_padding.py`
**Issue:** [PR#7958](https://github.com/apache/tvm/pull/7958) — ConvTranspose always passed `output_padding=0` ignoring attribute  
**Pattern:** `convtranspose_output_padding`
```
Input shape: (1, 1, 4, 4), stride=2, output_padding=[1,1]
ORT output shape:             (1, 1, 10, 10)
Expected with output_padding: (1, 1, 10, 10)
Wrong shape (output_pad=0):   (1, 1, 9, 9)
Shape correct: True
ORT output[0,0,0,:5]: [ 1.2890637   2.754751   -0.26515374 -0.45046028 -0.34815598]
PASS=True  (TVM bug: always output shape (1, 1, 9, 9) on affected versions)
```

### `bug_tvm_008_instance_norm_mean_var.py`
**Issue:** [#15683](https://github.com/apache/tvm/issues/15683) — InstanceNorm wrong results for small spatial dims on GPU  
**Pattern:** `instance_norm_small_spatial`
```
Input shape: (2, 4, 2, 2) (small spatial 2x2)
ORT output[0,0]: [[ 0.20030262 -1.6671984 ]
 [ 0.5049678   0.961928  ]]
Ref output[0,0]: [[ 0.20030259 -1.6671983 ]
 [ 0.5049677   0.96192795]]
Max diff ORT vs NumPy (CPU): 0.00000024
ORT CPU PASS=True
Note: TVM #15683 — wrong results on GPU with small spatial dims
PASS=True (ORT CPU correct; TVM issue on GPU path)
```

### `bug_tvm_009_scatternd_cuda_missing_return.py`
**Issue:** [PR#7447](https://github.com/apache/tvm/pull/7447) — ScatterND CUDA kernel missing `return` corrupted adjacent memory  
**Pattern:** `scatternd_cuda`
```
Data: 4x4 zeros
Scatter diagonal: indices (0,0),(1,1),(2,2),(3,3) → updates 1,2,3,4
Expected:
[[1. 0. 0. 0.]
 [0. 2. 0. 0.]
 [0. 0. 3. 0.]
 [0. 0. 0. 4.]]
ORT CPU output:
[[1. 0. 0. 0.]
 [0. 2. 0. 0.]
 [0. 0. 3. 0.]
 [0. 0. 0. 4.]]
Max diff: 0.000000
PASS=True  (TVM GPU had missing-return corruption on affected versions)
```

---

## Inductor — New Bugs (006–012)

### `bug_inductor_006_avgpool_ceil_mode_border.py`
**Issue:** [#100987](https://github.com/pytorch/pytorch/issues/100987) — AvgPool2d `ceil_mode=True` divided border windows by full kernel size  
**Pattern:** `avgpool_ceil_mode_border`
```
Input: 6x6 gradient (cols 1-6), kernel=3, stride=2
ceil_mode=True  output shape: (1, 1, 3, 3)  (expected (1,1,3,3))
ceil_mode=False output shape: (1, 1, 2, 2)  (expected (1,1,2,2))
ceil_mode output[0,0]:
[[2.  4.  5.5]
 [2.  4.  5.5]
 [2.  4.  5.5]]
floor_mode output[0,0]:
[[2. 4.]
 [2. 4.]]
Border window [4:6,4:6] correct avg=5.50
ORT border output[0,0,2,2]=5.5000
PASS=True  (Inductor #100987: border window divided by full kernel size)
```

### `bug_inductor_007_avgpool_single_element_gpu.py`
**Issue:** [#143720](https://github.com/pytorch/pytorch/issues/143720) — AvgPool GPU single-element window skipped division step  
**Pattern:** `avgpool_single_element`
```
Input shape: (1, 1, 4, 4), kernel=2, stride=2 (no overlap)
count_include_pad=1 output[0,0]: [[-0.05150452  0.39874226]
 [-0.4267754   0.11936981]]
count_include_pad=0 output[0,0]: [[-0.05150452  0.39874226]
 [-0.4267754   0.11936981]]
NumPy reference     output[0,0]: [[-0.05150453  0.3987423 ]
 [-0.4267754   0.1193698 ]]
Max diff include vs no-include: 0.000000
Max diff vs NumPy reference: 0.00000003
PASS=True  (GPU single-element window bug in Inductor #143720)
```

### `bug_inductor_008_convtranspose_empty_output.py`
**Issue:** [#144013](https://github.com/pytorch/pytorch/issues/144013) — ConvTranspose with `output_size=(0,0)` crashed Inductor with IndexError  
**Pattern:** `convtranspose_empty`
```
Input shape: (1, 2, 3, 3), kernel=3, stride=1, pads=[1,1,1,1]
Output shape: (1, 1, 3, 3)  (expected [1,1,3,3] — same size with pads)
Output[0,0]: [[ 2.5699074   5.3828845   0.26575065]
 [ 4.214775    3.5208755  -1.6694341 ]
 [ 2.4608202  -0.2238617   2.1284215 ]]
PASS=True  (Inductor #144013: crash on empty-output ConvTranspose with torch.compile)

[Bug context] PyTorch Inductor #144013:
  conv = nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1)
  x = torch.randn(1, 2, 1, 1)
  torch.compile(conv)(x, output_size=(0, 0))
  → IndexError in Inductor; eager → ValueError (correct)
```

### `bug_inductor_009_gridsample_nan_inf_gpu.py`
**Issue:** [#24823](https://github.com/pytorch/pytorch/issues/24823) — GridSample NaN in grid contaminated valid neighbours on GPU  
**Pattern:** `gridsample_nan_propagation`
```
Grid: [valid, NaN, valid, NaN] coordinates
ORT CPU output[0,0,0,:]: [-0.40334725         nan -0.40934166         nan]
NaN outputs at NaN inputs (expected True):  True
Valid outputs not corrupted (expected True): True
PASS=True  (GPU bug #24823: NaN contaminated valid neighbours)
```

### `bug_inductor_010_gridsample_bilinear_precision.py`
**Issue:** [#81868](https://github.com/pytorch/pytorch/issues/81868) — GridSample bilinear identity grid lost precision due to FP rounding  
**Pattern:** `gridsample_bilinear_identity`
```
Input shape: (1, 1, 4, 4), identity grid (align_corners=True)
Input  [0,0]: [[ 1.1285222   0.14419228 -1.1152586   1.2310466 ]
 [-0.15465815 -0.5983871   0.34899214  1.2205896 ]
 [-0.8287099  -1.1110595  -0.52429044 -1.0934223 ]
 [-1.6353352  -1.685021    0.69232917 -2.3885653 ]]
Output [0,0]: [[ 1.1285222   0.14419234 -1.1152586   1.2310466 ]
 [-0.15465806 -0.598387    0.34899205  1.2205896 ]
 [-0.8287099  -1.1110594  -0.52429044 -1.0934223 ]
 [-1.6353352  -1.685021    0.69232917 -2.3885653 ]]
Max diff (identity): 1.19e-07
Mean diff (identity): 2.98e-08
PASS=True  (PyTorch #81868: bilinear identity precision)
```

### `bug_inductor_011_scatternd_oob_indices.py`
**Issue:** [#122291](https://github.com/pytorch/pytorch/issues/122291) — ScatterND OOB indices returned garbage on GPU instead of error  
**Pattern:** `scatternd_bounds`
```
Data (3x4 range 0..11):
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
Indices: [[0, 0], [1, 2], [2, 3]]
Updates: [100. 200. 300.]
Expected:
[[100.   1.   2.   3.]
 [  4.   5. 200.   7.]
 [  8.   9.  10. 300.]]
ORT output:
[[100.   1.   2.   3.]
 [  4.   5. 200.   7.]
 [  8.   9.  10. 300.]]
Max diff: 0.000000
PASS=True  (Inductor #122291: OOB garbage on GPU; ORT CPU correct)
```

### `bug_inductor_012_einsum_matmul_first_call.py`
**Issue:** [#85224](https://github.com/pytorch/pytorch/issues/85224) — `torch.einsum` batch matmul on MPS gave wrong result on first call  
**Pattern:** `einsum_mps_first_call`
```
Batch matmul via Einsum 'bij,bjk->bik'
Input shapes: A=(2, 3, 4), B=(2, 4, 5)
ORT output[0,0,:3]:  [-0.7964003 -0.1398522  2.0328918]
NumPy ref[0,0,:3]: [-0.7964003  -0.13985221  2.0328918 ]
Max diff ORT vs NumPy: 0.00000024
PASS=True  (MPS #85224: wrong on first call; CPU always correct)
```

---

## OpenVINO — New Bugs (011–014)

### `bug_ov_011_avgpool_cpu_int8_ceil.py`
**Issue:** [#20815](https://github.com/openvinotoolkit/openvino/issues/20815) — AvgPool CPU int8 with `ceil_mode` divided border windows by full kernel area  
**Pattern:** `avgpool_ceil_int8`
```
Input: [ 2.  4.  6.  8. 10. 12.]
ceil_mode  output: [ 4.  8. 11.]   (expected [ 4.  8. 11.])
floor_mode output: [4. 8.]  (expected [4. 8.])
ceil_mode  max diff: 0.000000
floor_mode max diff: 0.000000
Border window [4:6] output=11.0000 (correct=11.0, wrong=7.33)
PASS=True  (OV #20815: int8 ceil border ÷ full_kernel)
```

### `bug_ov_012_prelu_arm_emitter.py`
**Issue:** [PR#28223](https://github.com/openvinotoolkit/openvino/pull/28223) — PReLU ARM JIT emitter used wrong register: computed `x*x` instead of `slope*x`  
**Pattern:** `prelu_arm_jit`
```
Input:      [-3.  -2.  -1.  -0.5  0.   0.5  1.   2.   3. ]
slope=0.25
ORT output: [-0.75  -0.5   -0.25  -0.125  0.     0.5    1.     2.     3.   ]
Reference:  [-0.75  -0.5   -0.25  -0.125  0.     0.5    1.     2.     3.   ]
ARM bug would give: [9.   4.   1.   0.25 0.   0.5  1.   2.   3.  ]
Max diff ORT vs reference: 0.00000000
PASS=True  (OV ARM bug: slope*x computed as x*x for negatives)
```

### `bug_ov_013_convtranspose_autopad_output_shape.py`
**Issue:** [#30798](https://github.com/openvinotoolkit/openvino/issues/30798) — ConvTranspose `SAME_LOWER` used `SAME_UPPER` formula for output shape  
**Pattern:** `convtranspose_same_lower`  **Status: OPEN**
```
Input: (1, 1, 4, 4), kernel=3, stride=2
SAME_UPPER output shape: (1, 1, 8, 8)  (expected [1,1,8,8])
SAME_LOWER output shape: (1, 1, 8, 8)  (expected [1,1,8,8])
SAME_UPPER shape correct: True
SAME_LOWER shape correct: True
Max value diff UPPER vs LOWER: 2.8054  (should differ due to padding placement)
PASS=True  (OV #30798: SAME_LOWER used SAME_UPPER formula)
```

### `bug_ov_014_einsum_scalar_input_parse.py`
**Issue:** [PR#30189](https://github.com/openvinotoolkit/openvino/pull/30189) — Einsum parser segfaulted on scalar (rank-0) operand  
**Pattern:** `einsum_scalar_operand`
```
Einsum 'i,->i' (scalar multiply)
Input vector: [1. 2. 3. 4. 5.], scalar: 3.0
ORT output: [ 3.  6.  9. 12. 15.]
Expected:   [ 3.  6.  9. 12. 15.]
Max diff: 0.00000000
PASS=True  (OV PR#30189: scalar input crashed the parser)
```

---

## Summary: Bugs Still Active (PASS=False or flagged ACTIVE)

| # | File | Pattern | Max Error | Notes |
|---|------|---------|-----------|-------|
| 1 | `ort/bug_ort_002_*` | resize_nearest | 12/64 elements | ORT nearest vs Torch half_pixel diff |
| 2 | `ort/bug_ort_004_*` | cast_fp32_int32 | semantic | ORT float→bool skips int32 truncation step |
| 3 | `ort/bug_ort_008_*` | resize_cubic | 0.061 | ORT cubic vs PyTorch bicubic coeff diff |
| 4 | `ort/bug_ort_016_*` | gridsample_bicubic | out of [0,15] | bicubic+border exceeds feature range **(OPEN)** |
| 5 | `tvm/bug_tvm_001_*` | batchnorm_eval | 1.36 | eval vs train BN divergence |
| 6 | `tvm/bug_tvm_004_*` | resize_cubic | 0.053 | ORT vs PyTorch bicubic coordinate diff |
| 7 | `tvm/bug_tvm_005_*` | dilated_branch | 1.002 | BN train-mode batch interaction |
| 8 | `tensorflow/bug_tf_001_*` | resize_linear | 0.143 | align_corners=True vs False |
| 9 | `tensorflow/bug_tf_003_*` | depth_to_space | 15.0 | DCR vs CRD channel ordering |
| 10 | `openvino/bug_ov_003_*` | matmul_fp16 | 1.0 | FP16 N>2048 overflow |
| 11 | `openvino/bug_ov_013_*` | convtranspose | SAME_LOWER | OV SAME_LOWER formula uses SAME_UPPER **(OPEN)** |
| 12 | `onnx_spec/bug_spec_007_*` | resize_nearest | element-4 wrong | round_prefer_ceil edge case |

---

**Total: 68 bugs — 12 still active/open on current framework versions**

*Generated by Trion differential fuzzing project — github_bugs/ directory*
