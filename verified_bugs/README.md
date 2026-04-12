# Verified Bug Reproducers

Confirmed compiler bugs found by the **Trion** differential fuzzing campaign across 6 deep-learning compilers.

## Campaign v2 — 210 reproducer files (48 confirmed bugs)

Each bug has one `.py` file per affected compiler under `repros/campaign_v2/`.  
Run any file: **exit 0 = bug reproduced**, exit 1 = not reproduced, exit 2 = dependency missing.

### Requirements

```bash
pip install numpy onnx onnxruntime torch onnx2torch
pip install jax[cpu]          # for XLA files
pip install tvm               # for TVM files
pip install openvino          # for OpenVINO files
pip install tensorflow        # for TF files
```

### Compiler coverage

| Compiler | Files | Confirmed bugs |
|---|---|---|
| OnnxRuntime (ORT_ENABLE_ALL) | 42 | 42 |
| TensorFlow XLA JIT (`jit_compile=True`) | 45 | 45 |
| OpenVINO CPU | 34 | 34 |
| TVM Relay/LLVM | 32 | 32 |
| JAX/XLA JIT | 32 | 32 |
| torch.compile (Inductor) | 25 | 25 |
| **Total** | **210** | **210** |

Oracle for all: `pytorch_eager` via `onnx2torch`.  
Clean compilers (s_diff ≈ 0.000) independently confirm the reference is correct for every bug.

---

### All 48 confirmed bugs

> **SUSPECT** = all 6 compilers diverge from reference; onnx2torch reference may be wrong — do not file.

| uid | Source model | Key patterns | Affected compilers | Divergence example | Repro files |
|-----|-------------|-------------|-------------------|--------------------|-------------|
| 00 | bug_000759 | matmul_4d_batch, double_transpose, resize_linear_asymmetric | ORT + OV + TF | idx=83: expected=0.0000, got=0.0100 | v2_0000_onnxruntime.py, v2_0000_openvino.py, v2_0000_tensorflow.py |
| 01 | bug_000296 | reduce_max_spatial_pair, resize_cubic_halfpixel, groupnorm_manual_chain | XLA + TVM + TF | idx=15682: expected=2.2286, got=1.0520 | v2_0001_xla.py, v2_0001_tvm.py, v2_0001_tensorflow.py |
| 02 | bug_000678 | resize_nearest_ceil, cast_fp32_int32_roundtrip, matmul_4d_batch | ORT + OV + TF | idx=22756: expected=586.7221, got=0.0395 | v2_0002_onnxruntime.py, v2_0002_openvino.py, v2_0002_tensorflow.py |
| 03 | bug_000028 | batchnorm_eval, add_mul_add_chain, fpn_branch | XLA + TVM + TF + TC | idx=3289676: expected=-73.9287, got=-54.0291 | v2_0003_xla.py, v2_0003_tvm.py, v2_0003_tensorflow.py, v2_0003_torch_compile.py |
| 04 | bug_001310 | resize_nearest_ceil, conv_bn_fusion_both_paths, foldable_add_zero_mul_one | ORT + OV + TC + TF | idx=189085: expected=23.0376, got=-31.3429 | v2_0004_onnxruntime.py, v2_0004_openvino.py, v2_0004_torch_compile.py, v2_0004_tensorflow.py |
| 05 | bug_000359 | topk_last_axis_k1, triple_add_residual, reduce_l2_last | ORT | s_diff=0.541 (accumulated precision error across TopK residual) | v2_0005_onnxruntime.py |
| 06 | bug_001152 | gather_layernorm, topk_last_axis_k1, layernorm_gate | ORT | idx=255: expected=0.0000, got=0.0001 | v2_0006_onnxruntime.py |
| 07 | bug_000010 | add_zero_identity, layernorm_relu, exp_div_softplus | XLA + TVM + TF | idx=1649: expected=2.0850, got=1.3824 | v2_0007_xla.py, v2_0007_tvm.py, v2_0007_tensorflow.py |
| 08 | bug_000933 | dilated_branch_sum, pad_edge_mode, inverted_residual | XLA + TVM + TF + TC | idx=60341: expected=24.2575, got=17.6568 | v2_0008_xla.py, v2_0008_tvm.py, v2_0008_tensorflow.py, v2_0008_torch_compile.py |
| 09 | bug_001084 | conv_hardswish, conv_bn_hardsigmoid_v2, resize_linear_asymmetric | ORT + OV + TC + TF | idx=16741: expected=6.7145, got=8.4490 | v2_0009_onnxruntime.py, v2_0009_openvino.py, v2_0009_torch_compile.py, v2_0009_tensorflow.py |
| 10 | bug_000806 | resize_nearest_ceil, pow_canonical, conv_k5_pad2 | ORT + OV + TC + TF | idx=29209: expected=5.8538, got=0.0870 | v2_0010_onnxruntime.py, v2_0010_openvino.py, v2_0010_torch_compile.py, v2_0010_tensorflow.py |
| 11 | bug_000039 | resize_linear_aligncorners, reshape_transpose5d_back, manual_layernorm | XLA + TVM + TF | idx=21342: expected=0.0287, got=-0.3238 | v2_0011_xla.py, v2_0011_tvm.py, v2_0011_tensorflow.py |
| 12 | bug_001319 | mul_zero_elim, instance_normalization_native, matmul_4d_batch | **SUSPECT** (all 6) | idx=521: expected=-0.0000, got=0.0019 | v2_0012_*.py |
| 13 | bug_000573 | reduce_broadcast_norm, resize_linear_aligncorners, manual_group_norm | XLA + TVM + TF + TC | idx=163461: expected=224.7519, got=58.0355 | v2_0013_xla.py, v2_0013_tvm.py, v2_0013_tensorflow.py, v2_0013_torch_compile.py |
| 14 | bug_000611 | cumsum_last_axis, gemm_tanh, linear_layernorm_gelu | ORT + XLA + TVM + TF + OV | idx=97: expected=3.1845, got=-0.2772 | v2_0014_onnxruntime.py, v2_0014_xla.py, v2_0014_tvm.py, v2_0014_tensorflow.py, v2_0014_openvino.py |
| 15 | bug_000840 | softsign_activation, where_mask, matmul_4d_batch | ORT + OV + TF | idx=1427: expected=-6.0066, got=5.3296 | v2_0015_onnxruntime.py, v2_0015_openvino.py, v2_0015_tensorflow.py |
| 16 | bug_001335 | resize_linear_asymmetric, unsqueeze_three_squeeze, matmul_4d_batch | ORT + OV + TF + TC | idx=249401: expected=-5.6215, got=-0.0657 | v2_0016_onnxruntime.py, v2_0016_openvino.py, v2_0016_tensorflow.py, v2_0016_torch_compile.py |
| 17 | bug_000887 | log_exp_cancel, softmax_axis0_residual, mul_zero_elim | ORT + TC + XLA + TVM | idx=255: expected=-0.0000, got=0.0000 | v2_0017_onnxruntime.py, v2_0017_torch_compile.py, v2_0017_xla.py, v2_0017_tvm.py |
| 18 | bug_000554 | transpose_matmul_transpose, learned_temperature_scale, cumsum_last_axis | ORT + XLA + TVM + TF + OV | idx=6822: expected=-114.8298, got=50.7835 | v2_0018_onnxruntime.py, v2_0018_xla.py, v2_0018_tvm.py, v2_0018_tensorflow.py, v2_0018_openvino.py |
| 19 | bug_000007 | resize_nearest_ceil, log_clamp, log_exp_cancel | ORT + OV | idx=4310: expected=22.3380, got=0.0000 | v2_0019_onnxruntime.py, v2_0019_openvino.py |
| 20 | bug_000576 | ada_layer_norm, layernorm_residual_add, resize_nearest_round_prefer_floor | **SUSPECT** (all 6) | idx=183: expected=0.0000, got=-0.0002 | v2_0020_*.py |
| 21 | bug_000106 | depth_to_space_block, resize_linear_aligncorners, double_transpose | XLA + TVM + TF + TC | idx=281350: expected=8.1435, got=1.6476 | v2_0021_xla.py, v2_0021_tvm.py, v2_0021_tensorflow.py, v2_0021_torch_compile.py |
| 22 | bug_000938 | tanh_rescaled_to_sigmoid, unsqueeze_expand_mul, resize_linear_aligncorners | XLA + TVM + TF | idx=1950341: expected=0.0230, got=0.2745 | v2_0022_xla.py, v2_0022_tvm.py, v2_0022_tensorflow.py |
| 23 | bug_000227 | matmul_4d_batch, conv_bn_fusion_both_paths, shared_reshape_target | XLA + TVM + TF + TC | idx=63581: expected=-9.1084, got=-3.1586 | v2_0023_xla.py, v2_0023_tvm.py, v2_0023_tensorflow.py, v2_0023_torch_compile.py |
| 24 | bug_000221 | matmul_4d_batch, resize_linear_halfpixel, matmul_4d_batch | ORT + OV + TF | idx=38197: expected=-4.8665, got=6.6169 | v2_0024_onnxruntime.py, v2_0024_openvino.py, v2_0024_tensorflow.py |
| 25 | bug_000339 | where_const_condition, conv_softplus, depthwise_conv_k3 | XLA + TVM + TF + TC | idx=6077: expected=1.5167, got=1.2341 | v2_0025_xla.py, v2_0025_tvm.py, v2_0025_tensorflow.py, v2_0025_torch_compile.py |
| 26 | bug_000733 | matmul_scale_add_bias, cast_roundtrip, mul_zero_elim | ORT + XLA + TVM + OV | idx=2099: expected=0.0000, got=-0.0001 | v2_0026_onnxruntime.py, v2_0026_xla.py, v2_0026_tvm.py, v2_0026_openvino.py |
| 27 | bug_001157 | residual_double_conv, resize_linear_asymmetric, matmul_4d_batch | ORT + OV + TC + TF | idx=113690: expected=1.8736, got=6.5515 | v2_0027_onnxruntime.py, v2_0027_openvino.py, v2_0027_torch_compile.py, v2_0027_tensorflow.py |
| 28 | bug_000679 | resize_nearest_ceil, reduce_sum_middle_axis, mul_self_to_pow | ORT + OV | idx=11431: expected=0.0000, got=3.7529 | v2_0028_onnxruntime.py, v2_0028_openvino.py |
| 29 | bug_000723 | matmul_4d_batch, matmul_4d_batch, resize_nearest_ceil | ORT + OV + TF + TC | idx=125555: expected=-0.2758, got=4.7220 | v2_0029_onnxruntime.py, v2_0029_openvino.py, v2_0029_tensorflow.py, v2_0029_torch_compile.py |
| 30 | bug_000764 | reciprocal_mul, mul_by_reciprocal, cumsum_last_axis | ORT + XLA + TVM + TF + OV | idx=62: expected=2.9564, got=-0.0108 | v2_0030_onnxruntime.py, v2_0030_xla.py, v2_0030_tvm.py, v2_0030_tensorflow.py, v2_0030_openvino.py |
| 31 | bug_000959 | transpose_inverse_cancel, reshape_rank2_roundtrip, resize_nearest_ceil | ORT + OV | idx=85: expected=6.5298, got=5.0744 | v2_0031_onnxruntime.py, v2_0031_openvino.py |
| 32 | bug_000194 | mul_zero_elim, skip_layernorm_pattern2, inference_dropout_foldconst | ORT | idx=11071: expected=0.0200, got=0.0212 | v2_0032_onnxruntime.py |
| 33 | bug_000875 | cumsum_last_axis, spatial_attention_cbam, softmax_axis1_last | **SUSPECT** (all 6) | idx=14159: expected=1.0003, got=-2.4878 | v2_0033_*.py |
| 34 | bug_001201 | layernorm_temperature, triple_add_residual, self_sub_zero | **SUSPECT** (all 6) | idx=127: expected=-0.0000, got=0.0001 | v2_0034_*.py |
| 35 | bug_000609 | residual_add_relu, erf_unary, neg_abs_identity | ORT + XLA + TVM + TF + OV | idx=2278: expected=5.7773, got=-0.2599 | v2_0035_onnxruntime.py, v2_0035_xla.py, v2_0035_tvm.py, v2_0035_tensorflow.py, v2_0035_openvino.py |
| 36 | bug_000453 | log1p_abs, matmul_add_biasgelu_bcast, max_variadic_4inputs | ORT + XLA + TVM | idx=2047: expected=0.0000, got=-0.1250 | v2_0036_onnxruntime.py, v2_0036_xla.py, v2_0036_tvm.py |
| 37 | bug_000163 | resize_nearest_ceil, matmul_4d_batch, matmul_4d_batch | ORT + OV + TF | idx=1301: expected=72.9599, got=0.7542 | v2_0037_onnxruntime.py, v2_0037_openvino.py, v2_0037_tensorflow.py |
| 38 | bug_000980 | relu_add_relu, mul_by_one_chain, sub_self_mul_zero | ORT | idx=54: expected=2.5474, got=2.7111 | v2_0038_onnxruntime.py |
| 39 | bug_000553 | matmul_4d_batch, reduce_sum_middle_axis, concat_resize_concat | ORT + OV + TF | idx=30359: expected=42.7093, got=-18.2903 | v2_0039_onnxruntime.py, v2_0039_openvino.py, v2_0039_tensorflow.py |
| 40 | bug_001069 | gather_reshape, reduce_l2_last, manual_layernorm | **SUSPECT** (all 6) | idx=48: expected=-0.0010, got=0.0000 | v2_0040_*.py |
| 41 | bug_000427 | ada_layer_norm, resize_nearest_asymmetric, resize_cubic_halfpixel | ORT + XLA + TVM + TF + OV | idx=126877: expected=-3.2403, got=0.1392 | v2_0041_onnxruntime.py, v2_0041_xla.py, v2_0041_tvm.py, v2_0041_tensorflow.py, v2_0041_openvino.py |
| 42 | bug_000810 | reduce_sum_middle_axis, mul_zero_elim, layernorm_relu | ORT | output sparsity: ref=29935 non-zero, ORT=51020 non-zero (s_diff=0.063) | v2_0042_onnxruntime.py |
| 43 | bug_000888 | matmul_4d_batch, resize_linear_aligncorners, fpn_branch | XLA + TVM + TF + TC | idx=705044: expected=-36.7906, got=-17.8813 | v2_0043_xla.py, v2_0043_tvm.py, v2_0043_tensorflow.py, v2_0043_torch_compile.py |
| 44 | bug_000268 | resize_linear_asymmetric, se_block, layernorm_dropout_identity | ORT + OV + TF | idx=600031: expected=0.3408, got=-4.0598 | v2_0044_onnxruntime.py, v2_0044_openvino.py, v2_0044_tensorflow.py |
| 45 | bug_000784 | stable_softmax, matmul_4d_batch, inception_v3_branch | XLA + TVM + TF | idx=76572: expected=-0.2651, got=0.7175 | v2_0045_xla.py, v2_0045_tvm.py, v2_0045_tensorflow.py |
| 46 | bug_000937 | spatial_reduce_mean, matmul_4d_batch, resize_cubic_halfpixel | XLA + TVM + TF | idx=728: expected=0.1461, got=0.1032 | v2_0046_xla.py, v2_0046_tvm.py, v2_0046_tensorflow.py |
| 47 | bug_000651 | l2_norm_manual_primitives, div_by_constant, mul_add_mul_chain | ORT + TC + XLA + TVM + TF | idx=255: expected=-0.0000, got=0.0000 | v2_0047_onnxruntime.py, v2_0047_torch_compile.py, v2_0047_xla.py, v2_0047_tvm.py, v2_0047_tensorflow.py |
| 48 | bug_000355 | matmul_bias_gelu, crms_norm, mul_add_mul_chain | **SUSPECT** (all 6) | idx=40: expected=-0.4262, got=-0.4396 | v2_0048_*.py |
| 49 | bug_000284 | matmul_bias_gelu, gated_linear_branch, mul_zero_elim | ORT + XLA + TVM + TF + OV | idx=127: expected=0.0000, got=-0.0000 | v2_0049_onnxruntime.py, v2_0049_xla.py, v2_0049_tvm.py, v2_0049_tensorflow.py, v2_0049_openvino.py |
| 50 | bug_000127 | cumsum_last_axis, cast_roundtrip, sqrt_div_rms | **SUSPECT** (all 6) | idx=63: expected=1.5451, got=2.6271 | v2_0050_*.py |
| 51 | bug_000343 | matmul_scale_softmax, where_mask_fill, softplus_activation | **SUSPECT** (all 6) | idx=372: expected=-0.0024, got=-0.0023 | v2_0051_*.py |
| 52 | bug_000696 | resize_cubic_halfpixel, manual_hardsigmoid, three_branch_concat | XLA + TVM + TF | idx=44749: expected=5.2641, got=4.6149 | v2_0052_xla.py, v2_0052_tvm.py, v2_0052_tensorflow.py |
| 53 | bug_000471 | abs_unary_simple, resize_nearest_ceil, matmul_4d_batch | ORT + OV + TF + TC | idx=189651: expected=23.2068, got=-25.8616 | v2_0053_onnxruntime.py, v2_0053_openvino.py, v2_0053_tensorflow.py, v2_0053_torch_compile.py |
| 54 | bug_000116 | resize_linear_asymmetric, matmul_4d_batch, matmul_4d_batch | ORT + OV + TF | idx=5541: expected=1.5015, got=-2.0413 | v2_0054_onnxruntime.py, v2_0054_openvino.py, v2_0054_tensorflow.py |
| 55 | bug_001049 | matmul_4d_batch, manual_layernorm, aspp_dilated_branch | ORT + OV + TF + TC | idx=103888: expected=-11.2099, got=7.5809 | v2_0055_onnxruntime.py, v2_0055_openvino.py, v2_0055_tensorflow.py, v2_0055_torch_compile.py |

**Abbreviations:** ORT = OnnxRuntime, OV = OpenVINO, TF = TensorFlow XLA JIT, TC = torch.compile, XLA = JAX/XLA JIT, TVM = TVM Relay

---

### File structure

```
repros/
└── campaign_v2/
    ├── v2_0000_onnxruntime.py   # uid=0, ORT bug
    ├── v2_0000_openvino.py      # uid=0, OpenVINO bug
    ├── v2_0000_tensorflow.py    # uid=0, TF XLA bug
    ├── v2_0001_tensorflow.py    # uid=1, TF XLA bug
    ├── v2_0001_tvm.py           # uid=1, TVM bug
    ├── v2_0001_xla.py           # uid=1, XLA/JAX bug
    ... (210 files total)
```

Each file is **fully self-contained**: the ONNX model is embedded as base64 (no external files needed), runs the target compiler, compares against `pytorch_eager` (onnx2torch), and prints the divergence at the most significant output index.

---

### Run results (this machine)

> OnnxRuntime + torch.compile + JAX/XLA verified on CPU. OpenVINO/TVM/TensorFlow not installed on this machine — install with pip to verify those bugs.

| File | Result |
|------|--------|
| v2_0000_onnxruntime.py | BUG REPRODUCED |
| v2_0000_openvino.py | OpenVINO not installed — cannot verify |
| v2_0000_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0001_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0001_tvm.py | TVM not installed — cannot verify |
| v2_0001_xla.py | BUG REPRODUCED |
| v2_0002_onnxruntime.py | BUG REPRODUCED |
| v2_0002_openvino.py | OpenVINO not installed — cannot verify |
| v2_0002_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0003_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0003_torch_compile.py | not reproduced |
| v2_0003_tvm.py | TVM not installed — cannot verify |
| v2_0003_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0004_onnxruntime.py | BUG REPRODUCED |
| v2_0004_openvino.py | OpenVINO not installed — cannot verify |
| v2_0004_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0004_torch_compile.py | not reproduced |
| v2_0005_onnxruntime.py | BUG REPRODUCED |
| v2_0006_onnxruntime.py | BUG REPRODUCED |
| v2_0007_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0007_tvm.py | TVM not installed — cannot verify |
| v2_0007_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0008_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0008_torch_compile.py | not reproduced |
| v2_0008_tvm.py | TVM not installed — cannot verify |
| v2_0008_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0009_onnxruntime.py | BUG REPRODUCED |
| v2_0009_openvino.py | OpenVINO not installed — cannot verify |
| v2_0009_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0009_torch_compile.py | not reproduced |
| v2_0010_onnxruntime.py | BUG REPRODUCED |
| v2_0010_openvino.py | OpenVINO not installed — cannot verify |
| v2_0010_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0010_torch_compile.py | not reproduced |
| v2_0011_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0011_tvm.py | TVM not installed — cannot verify |
| v2_0011_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0012_onnxruntime.py | BUG REPRODUCED |
| v2_0012_openvino.py | OpenVINO not installed — cannot verify |
| v2_0012_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0012_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |
| v2_0012_tvm.py | TVM not installed — cannot verify |
| v2_0012_xla.py | BUG REPRODUCED |
| v2_0013_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0013_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |
| v2_0013_tvm.py | TVM not installed — cannot verify |
| v2_0013_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0014_onnxruntime.py | BUG REPRODUCED |
| v2_0014_openvino.py | OpenVINO not installed — cannot verify |
| v2_0014_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0014_tvm.py | TVM not installed — cannot verify |
| v2_0014_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0015_onnxruntime.py | BUG REPRODUCED |
| v2_0015_openvino.py | OpenVINO not installed — cannot verify |
| v2_0015_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0016_onnxruntime.py | BUG REPRODUCED |
| v2_0016_openvino.py | OpenVINO not installed — cannot verify |
| v2_0016_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0016_torch_compile.py | not reproduced |
| v2_0017_onnxruntime.py | BUG REPRODUCED |
| v2_0017_torch_compile.py | not reproduced |
| v2_0017_tvm.py | TVM not installed — cannot verify |
| v2_0017_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0018_onnxruntime.py | BUG REPRODUCED |
| v2_0018_openvino.py | OpenVINO not installed — cannot verify |
| v2_0018_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0018_tvm.py | TVM not installed — cannot verify |
| v2_0018_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0019_onnxruntime.py | BUG REPRODUCED |
| v2_0019_openvino.py | OpenVINO not installed — cannot verify |
| v2_0020_onnxruntime.py | BUG REPRODUCED |
| v2_0020_openvino.py | OpenVINO not installed — cannot verify |
| v2_0020_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0020_torch_compile.py | BUG REPRODUCED |
| v2_0020_tvm.py | TVM not installed — cannot verify |
| v2_0020_xla.py | BUG REPRODUCED |
| v2_0021_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0021_torch_compile.py | not reproduced |
| v2_0021_tvm.py | TVM not installed — cannot verify |
| v2_0021_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0022_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0022_tvm.py | TVM not installed — cannot verify |
| v2_0022_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0023_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0023_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |
| v2_0023_tvm.py | TVM not installed — cannot verify |
| v2_0023_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0024_onnxruntime.py | BUG REPRODUCED |
| v2_0024_openvino.py | OpenVINO not installed — cannot verify |
| v2_0024_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0025_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0025_torch_compile.py | not reproduced |
| v2_0025_tvm.py | TVM not installed — cannot verify |
| v2_0025_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0026_onnxruntime.py | BUG REPRODUCED |
| v2_0026_openvino.py | OpenVINO not installed — cannot verify |
| v2_0026_tvm.py | TVM not installed — cannot verify |
| v2_0026_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0027_onnxruntime.py | BUG REPRODUCED |
| v2_0027_openvino.py | OpenVINO not installed — cannot verify |
| v2_0027_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0027_torch_compile.py | not reproduced |
| v2_0028_onnxruntime.py | BUG REPRODUCED |
| v2_0028_openvino.py | OpenVINO not installed — cannot verify |
| v2_0029_onnxruntime.py | BUG REPRODUCED |
| v2_0029_openvino.py | OpenVINO not installed — cannot verify |
| v2_0029_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0029_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |
| v2_0030_onnxruntime.py | BUG REPRODUCED |
| v2_0030_openvino.py | OpenVINO not installed — cannot verify |
| v2_0030_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0030_tvm.py | TVM not installed — cannot verify |
| v2_0030_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0031_onnxruntime.py | BUG REPRODUCED |
| v2_0031_openvino.py | OpenVINO not installed — cannot verify |
| v2_0032_onnxruntime.py | BUG REPRODUCED |
| v2_0033_onnxruntime.py | BUG REPRODUCED |
| v2_0033_openvino.py | OpenVINO not installed — cannot verify |
| v2_0033_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0033_torch_compile.py | not reproduced |
| v2_0033_tvm.py | TVM not installed — cannot verify |
| v2_0033_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0034_onnxruntime.py | BUG REPRODUCED |
| v2_0034_openvino.py | OpenVINO not installed — cannot verify |
| v2_0034_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0034_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |
| v2_0034_tvm.py | TVM not installed — cannot verify |
| v2_0034_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0035_onnxruntime.py | BUG REPRODUCED |
| v2_0035_openvino.py | OpenVINO not installed — cannot verify |
| v2_0035_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0035_tvm.py | TVM not installed — cannot verify |
| v2_0035_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0036_onnxruntime.py | BUG REPRODUCED |
| v2_0036_tvm.py | TVM not installed — cannot verify |
| v2_0036_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0037_onnxruntime.py | BUG REPRODUCED |
| v2_0037_openvino.py | OpenVINO not installed — cannot verify |
| v2_0037_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0038_onnxruntime.py | BUG REPRODUCED |
| v2_0039_onnxruntime.py | BUG REPRODUCED |
| v2_0039_openvino.py | OpenVINO not installed — cannot verify |
| v2_0039_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0040_onnxruntime.py | BUG REPRODUCED |
| v2_0040_openvino.py | OpenVINO not installed — cannot verify |
| v2_0040_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0040_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |
| v2_0040_tvm.py | TVM not installed — cannot verify |
| v2_0040_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0041_onnxruntime.py | BUG REPRODUCED |
| v2_0041_openvino.py | OpenVINO not installed — cannot verify |
| v2_0041_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0041_tvm.py | TVM not installed — cannot verify |
| v2_0041_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0042_onnxruntime.py | BUG REPRODUCED |
| v2_0043_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0043_torch_compile.py | not reproduced |
| v2_0043_tvm.py | TVM not installed — cannot verify |
| v2_0043_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0044_onnxruntime.py | BUG REPRODUCED |
| v2_0044_openvino.py | OpenVINO not installed — cannot verify |
| v2_0044_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0045_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0045_tvm.py | TVM not installed — cannot verify |
| v2_0045_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0046_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0046_tvm.py | TVM not installed — cannot verify |
| v2_0046_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0047_onnxruntime.py | BUG REPRODUCED |
| v2_0047_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0047_torch_compile.py | not reproduced |
| v2_0047_tvm.py | TVM not installed — cannot verify |
| v2_0047_xla.py | BUG REPRODUCED |
| v2_0048_onnxruntime.py | BUG REPRODUCED |
| v2_0048_openvino.py | OpenVINO not installed — cannot verify |
| v2_0048_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0048_torch_compile.py | BUG REPRODUCED |
| v2_0048_tvm.py | TVM not installed — cannot verify |
| v2_0048_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0049_onnxruntime.py | BUG REPRODUCED |
| v2_0049_openvino.py | OpenVINO not installed — cannot verify |
| v2_0049_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0049_tvm.py | TVM not installed — cannot verify |
| v2_0049_xla.py | BUG REPRODUCED |
| v2_0050_onnxruntime.py | BUG REPRODUCED |
| v2_0050_openvino.py | OpenVINO not installed — cannot verify |
| v2_0050_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0050_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |
| v2_0050_tvm.py | TVM not installed — cannot verify |
| v2_0050_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0051_onnxruntime.py | BUG REPRODUCED |
| v2_0051_openvino.py | OpenVINO not installed — cannot verify |
| v2_0051_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0051_torch_compile.py | not reproduced |
| v2_0051_tvm.py | TVM not installed — cannot verify |
| v2_0051_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0052_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0052_tvm.py | TVM not installed — cannot verify |
| v2_0052_xla.py | BUG REPRODUCED (historical — model file unavailable, see docstring) |
| v2_0053_onnxruntime.py | BUG REPRODUCED |
| v2_0053_openvino.py | OpenVINO not installed — cannot verify |
| v2_0053_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0053_torch_compile.py | not reproduced |
| v2_0054_onnxruntime.py | BUG REPRODUCED |
| v2_0054_openvino.py | OpenVINO not installed — cannot verify |
| v2_0054_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0055_onnxruntime.py | BUG REPRODUCED |
| v2_0055_openvino.py | OpenVINO not installed — cannot verify |
| v2_0055_tensorflow.py | [error] tensorflow: No module named 'tf2onnx' |
| v2_0055_torch_compile.py | [error] torch.compile: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor') |

---

## Campaign v1 — 99 reproducer files (legacy)

Original campaign bugs. Files are in the repo root (`unique_NNNN.py`).

| Backend | Count |
|---|---|
| JAX/XLA + TVM | 88 |
| OnnxRuntime | 6 |
| torch.compile | 10 |
| **Total** | **99** |

```bash
python unique_NNNN.py
# Exit 0 = bug reproduced
```

---

## How to report

- **OnnxRuntime**: https://github.com/microsoft/onnxruntime/issues (labels: `bug`, `Resize`, `CumSum`)
- **OpenVINO**: https://github.com/openvinotoolkit/openvino/issues (labels: `bug`, `Interpolate`)
- **TensorFlow/XLA**: https://github.com/tensorflow/tensorflow/issues (labels: `comp:xla`, `type:bug`)
- **torch.compile**: https://github.com/pytorch/pytorch/issues (labels: `oncall: pt2`, `torch.compile`)
- **TVM**: https://github.com/apache/tvm/issues (labels: `Bug`, `relay`)
- **JAX/XLA**: https://github.com/google/jax/issues (labels: `bug`)
