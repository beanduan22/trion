# Reproducible Bugs — Trion Campaign v2

**48 confirmed compiler bugs** extracted from the Trion differential fuzzing campaign.

SUSPECT bugs (all 6 compilers diverge from the `onnx2torch` pytorch_eager reference — likely a reference error, not a real compiler bug) are **excluded**: uid 12, 20, 33, 34, 40, 48, 50, 51.

Each `.py` file is fully self-contained: the ONNX model is embedded as base64, no external files needed.  
Exit 0 = bug reproduced, exit 1 = not reproduced, exit 2 = dependency missing.

```bash
pip install numpy onnx onnxruntime torch onnx2torch
pip install jax[cpu]      # for XLA bugs
pip install tvm           # for TVM bugs
pip install openvino      # for OpenVINO bugs
pip install tensorflow    # for TF bugs
```

---

## Bug Index

| File | uid | Affected Compilers | Key Patterns | Divergence Example |
|------|-----|--------------------|--------------|-------------------|
| bug_v2_0000.py | 00 | ORT + OV + TF | matmul_4d_batch, double_transpose, resize_linear_asymmetric | idx=83: expected=0.0000, got=0.0100 |
| bug_v2_0001.py | 01 | XLA + TVM + TF | reduce_max_spatial_pair, resize_cubic_halfpixel, groupnorm_manual_chain | idx=15682: expected=2.2286, got=1.0520 |
| bug_v2_0002.py | 02 | ORT + OV + TF | resize_nearest_ceil, cast_fp32_int32_roundtrip, matmul_4d_batch | idx=22756: expected=586.7221, got=0.0395 |
| bug_v2_0003.py | 03 | XLA + TVM + TF + TC | batchnorm_eval, add_mul_add_chain, fpn_branch | idx=3289676: expected=-73.9287, got=-54.0291 |
| bug_v2_0004.py | 04 | ORT + OV + TC + TF | resize_nearest_ceil, conv_bn_fusion_both_paths, foldable_add_zero_mul_one | idx=189085: expected=23.0376, got=-31.3429 |
| bug_v2_0005.py | 05 | ORT | topk_last_axis_k1, triple_add_residual, reduce_l2_last | s_diff=0.541 (accumulated TopK precision) |
| bug_v2_0006.py | 06 | ORT | gather_layernorm, topk_last_axis_k1, layernorm_gate | idx=255: expected=0.0000, got=0.0001 |
| bug_v2_0007.py | 07 | XLA + TVM + TF | add_zero_identity, layernorm_relu, exp_div_softplus | idx=1649: expected=2.0850, got=1.3824 |
| bug_v2_0008.py | 08 | XLA + TVM + TF + TC | dilated_branch_sum, pad_edge_mode, inverted_residual | idx=60341: expected=24.2575, got=17.6568 |
| bug_v2_0009.py | 09 | ORT + OV + TC + TF | conv_hardswish, conv_bn_hardsigmoid_v2, resize_linear_asymmetric | idx=16741: expected=6.7145, got=8.4490 |
| bug_v2_0010.py | 10 | ORT + OV + TC + TF | resize_nearest_ceil, pow_canonical, conv_k5_pad2 | idx=29209: expected=5.8538, got=0.0870 |
| bug_v2_0011.py | 11 | XLA + TVM + TF | resize_linear_aligncorners, reshape_transpose5d_back, manual_layernorm | idx=21342: expected=0.0287, got=-0.3238 |
| bug_v2_0013.py | 13 | XLA + TVM + TF + TC | reduce_broadcast_norm, resize_linear_aligncorners, manual_group_norm | idx=163461: expected=224.7519, got=58.0355 |
| bug_v2_0014.py | 14 | ORT + XLA + TVM + TF + OV | cumsum_last_axis, gemm_tanh, linear_layernorm_gelu | idx=97: expected=3.1845, got=-0.2772 |
| bug_v2_0015.py | 15 | ORT + OV + TF | softsign_activation, where_mask, matmul_4d_batch | idx=1427: expected=-6.0066, got=5.3296 |
| bug_v2_0016.py | 16 | ORT + OV + TF + TC | resize_linear_asymmetric, unsqueeze_three_squeeze, matmul_4d_batch | idx=249401: expected=-5.6215, got=-0.0657 |
| bug_v2_0017.py | 17 | ORT + TC + XLA + TVM | log_exp_cancel, softmax_axis0_residual, mul_zero_elim | idx=255: expected=-0.0000, got=0.0000 |
| bug_v2_0018.py | 18 | ORT + XLA + TVM + TF + OV | transpose_matmul_transpose, learned_temperature_scale, cumsum_last_axis | idx=6822: expected=-114.8298, got=50.7835 |
| bug_v2_0019.py | 19 | ORT + OV | resize_nearest_ceil, log_clamp, log_exp_cancel | idx=4310: expected=22.3380, got=0.0000 |
| bug_v2_0021.py | 21 | XLA + TVM + TF + TC | depth_to_space_block, resize_linear_aligncorners, double_transpose | idx=281350: expected=8.1435, got=1.6476 |
| bug_v2_0022.py | 22 | XLA + TVM + TF | tanh_rescaled_to_sigmoid, unsqueeze_expand_mul, resize_linear_aligncorners | idx=1950341: expected=0.0230, got=0.2745 |
| bug_v2_0023.py | 23 | XLA + TVM + TF + TC | matmul_4d_batch, conv_bn_fusion_both_paths, shared_reshape_target | idx=63581: expected=-9.1084, got=-3.1586 |
| bug_v2_0024.py | 24 | ORT + OV + TF | matmul_4d_batch, resize_linear_halfpixel, matmul_4d_batch | idx=38197: expected=-4.8665, got=6.6169 |
| bug_v2_0025.py | 25 | XLA + TVM + TF + TC | where_const_condition, conv_softplus, depthwise_conv_k3 | idx=6077: expected=1.5167, got=1.2341 |
| bug_v2_0026.py | 26 | ORT + XLA + TVM + OV | matmul_scale_add_bias, cast_roundtrip, mul_zero_elim | idx=2099: expected=0.0000, got=-0.0001 |
| bug_v2_0027.py | 27 | ORT + OV + TC + TF | residual_double_conv, resize_linear_asymmetric, matmul_4d_batch | idx=113690: expected=1.8736, got=6.5515 |
| bug_v2_0028.py | 28 | ORT + OV | resize_nearest_ceil, reduce_sum_middle_axis, mul_self_to_pow | idx=11431: expected=0.0000, got=3.7529 |
| bug_v2_0029.py | 29 | ORT + OV + TF + TC | matmul_4d_batch, matmul_4d_batch, resize_nearest_ceil | idx=125555: expected=-0.2758, got=4.7220 |
| bug_v2_0030.py | 30 | ORT + XLA + TVM + TF + OV | reciprocal_mul, mul_by_reciprocal, cumsum_last_axis | idx=62: expected=2.9564, got=-0.0108 |
| bug_v2_0031.py | 31 | ORT + OV | transpose_inverse_cancel, reshape_rank2_roundtrip, resize_nearest_ceil | idx=85: expected=6.5298, got=5.0744 |
| bug_v2_0032.py | 32 | ORT | mul_zero_elim, skip_layernorm_pattern2, inference_dropout_foldconst | idx=11071: expected=0.0200, got=0.0212 |
| bug_v2_0035.py | 35 | ORT + XLA + TVM + TF + OV | residual_add_relu, erf_unary, neg_abs_identity | idx=2278: expected=5.7773, got=-0.2599 |
| bug_v2_0036.py | 36 | ORT + XLA + TVM | log1p_abs, matmul_add_biasgelu_bcast, max_variadic_4inputs | idx=2047: expected=0.0000, got=-0.1250 |
| bug_v2_0037.py | 37 | ORT + OV + TF | resize_nearest_ceil, matmul_4d_batch, matmul_4d_batch | idx=1301: expected=72.9599, got=0.7542 |
| bug_v2_0038.py | 38 | ORT | relu_add_relu, mul_by_one_chain, sub_self_mul_zero | idx=54: expected=2.5474, got=2.7111 |
| bug_v2_0039.py | 39 | ORT + OV + TF | matmul_4d_batch, reduce_sum_middle_axis, concat_resize_concat | idx=30359: expected=42.7093, got=-18.2903 |
| bug_v2_0041.py | 41 | ORT + XLA + TVM + TF + OV | ada_layer_norm, resize_nearest_asymmetric, resize_cubic_halfpixel | idx=126877: expected=-3.2403, got=0.1392 |
| bug_v2_0042.py | 42 | ORT | reduce_sum_middle_axis, mul_zero_elim, layernorm_relu | ref=29935 non-zero, ORT=51020 non-zero (s_diff=0.063) |
| bug_v2_0043.py | 43 | XLA + TVM + TF + TC | matmul_4d_batch, resize_linear_aligncorners, fpn_branch | idx=705044: expected=-36.7906, got=-17.8813 |
| bug_v2_0044.py | 44 | ORT + OV + TF | resize_linear_asymmetric, se_block, layernorm_dropout_identity | idx=600031: expected=0.3408, got=-4.0598 |
| bug_v2_0045.py | 45 | XLA + TVM + TF | stable_softmax, matmul_4d_batch, inception_v3_branch | idx=76572: expected=-0.2651, got=0.7175 |
| bug_v2_0046.py | 46 | XLA + TVM + TF | spatial_reduce_mean, matmul_4d_batch, resize_cubic_halfpixel | idx=728: expected=0.1461, got=0.1032 |
| bug_v2_0047.py | 47 | ORT + TC + XLA + TVM + TF | l2_norm_manual_primitives, div_by_constant, mul_add_mul_chain | idx=255: expected=-0.0000, got=0.0000 |
| bug_v2_0049.py | 49 | ORT + XLA + TVM + TF + OV | matmul_bias_gelu, gated_linear_branch, mul_zero_elim | idx=127: expected=0.0000, got=-0.0000 |
| bug_v2_0052.py | 52 | XLA + TVM + TF | resize_cubic_halfpixel, manual_hardsigmoid, three_branch_concat | idx=44749: expected=5.2641, got=4.6149 |
| bug_v2_0053.py | 53 | ORT + OV + TF + TC | abs_unary_simple, resize_nearest_ceil, matmul_4d_batch | idx=189651: expected=23.2068, got=-25.8616 |
| bug_v2_0054.py | 54 | ORT + OV + TF | resize_linear_asymmetric, matmul_4d_batch, matmul_4d_batch | idx=5541: expected=1.5015, got=-2.0413 |
| bug_v2_0055.py | 55 | ORT + OV + TF + TC | matmul_4d_batch, manual_layernorm, aspp_dilated_branch | idx=103888: expected=-11.2099, got=7.5809 |

**Abbreviations:** ORT = OnnxRuntime, OV = OpenVINO, TF = TensorFlow XLA JIT, TC = torch.compile, XLA = JAX/XLA JIT, TVM = TVM Relay

---

## Excluded (SUSPECT)

These 8 bugs were removed because **all 6 compilers** diverge from the `onnx2torch` reference — the reference itself is likely incorrect.

| uid | Source model | Key patterns |
|-----|-------------|--------------|
| 12 | bug_001319 | mul_zero_elim, instance_normalization_native, matmul_4d_batch |
| 20 | bug_000576 | ada_layer_norm, layernorm_residual_add, resize_nearest_round_prefer_floor |
| 33 | bug_000875 | cumsum_last_axis, spatial_attention_cbam, softmax_axis1_last |
| 34 | bug_001201 | layernorm_temperature, triple_add_residual, self_sub_zero |
| 40 | bug_001069 | gather_reshape, reduce_l2_last, manual_layernorm |
| 48 | bug_000355 | matmul_bias_gelu, crms_norm, mul_add_mul_chain |
| 50 | bug_000127 | cumsum_last_axis, cast_roundtrip, sqrt_div_rms |
| 51 | bug_000343 | matmul_scale_softmax, where_mask_fill, softplus_activation |

---

## Compiler Coverage Summary

| Compiler | Bugs |
|----------|------|
| OnnxRuntime (ORT) | 34 |
| OpenVINO (OV) | 26 |
| TensorFlow XLA JIT (TF) | 37 |
| torch.compile / Inductor (TC) | 17 |
| JAX/XLA JIT (XLA) | 24 |
| TVM Relay (TVM) | 24 |

(Counts include bugs affecting multiple compilers.)

## Root Cause Categories

| Category | Representative bugs |
|----------|-------------------|
| Resize coordinate mode (nearest ceil, linear asymmetric, cubic halfpixel, aligncorners) | 00, 02, 04, 09, 10, 11, 19, 21, 22, 27, 28, 31, 37, 41, 44, 52, 53, 54 |
| Algebraic elimination bugs (mul_zero, log_exp cancel, identity elim) | 17, 19, 26, 32, 36, 38, 47 |
| CumSum precision / axis | 14, 18, 30 |
| TopK tie-breaking / stability | 05, 06 |
| Conv+BN fusion / optimizer interaction | 04, 09, 23, 27 |
| MatMul 4D batch layout | 00, 15, 16, 24, 29, 37, 39, 43, 55 |
| Normalization (LayerNorm, BatchNorm, GroupNorm, InstanceNorm) | 03, 07, 13, 32, 42 |
| Misc (erf, softsign, depth-to-space, gelu, l2norm) | 01, 08, 25, 35, 41, 44, 45, 46, 49 |
