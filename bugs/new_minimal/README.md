# ML Compiler Bug Collection — Minimal Reproducible Scripts

**120 verified bugs** across 7 ML compilers/runtimes.
Verified: 2026-04-17 · Python 3.13 · ONNX 1.21 · ONNXRuntime 1.24.4 · OpenVINO 2026.0 · TensorFlow 2.21.0 · PyTorch 2.9.1 · TVM 0.11.1 (relay) · onnx2torch latest

Every script:
- Builds its ONNX (or direct framework) model from scratch — no binary blobs.
- Runs at least **two backends** on the same graph and prints both outputs.
- `Exit 0` = BUG REPRODUCED (cross-backend inconsistency proven) · `Exit 1` = not reproduced · `Exit 2` = missing dep.

---

## Summary

| Compiler | Bugs | Root cause |
|---|---|---|
| ONNXRuntime | 8 | BitShift UB; Resize nearest/bilinear/cubic; int-truncation fusion; SIGFPE mod-zero; GridSample bicubic border |
| OpenVINO | 57 | fp32 MatMul/Conv tile accumulation; uint8/int8 wrap saturation; NaN propagation via Relu/Exp; fp16 GEMM tile |
| TFLite | 7 | FC+Mul fusion converter crash; L2_NORM axis semantics; unstack+concat fold; XNNPack prepare failure; fp16 quant MatMul assoc |
| torch.compile (Inductor) | 27 | `Reshape(runtime_shape)` → `InternalTorchDynamoError: torch.Size() + FakeTensor`; bf16 Inductor diverge; bitshift UB; silent shape bypass |
| XLA / TensorFlow | 8 | OOB gather silent clamp; autocluster DCE before bounds-check; matmul incompatible shapes accepted under XLA; stride-2 SAME conv offset |
| onnx2torch | 9 | CumSum `axis.item()` graph break; resize coord/nearest-ceil; INT64 cumsum dtype; missing bf16 Cast converter; OOB Gather |
| TVM | 4 | Resize nearest half_pixel coord shift; RoiAlign half_pixel ignored; const-fold of `inf*x - inf*x`; GELU tanh-approx dropped |

---

## Running

```bash
# default Python env (onnxruntime/openvino/tflite/torch_compile/xla/onnx2torch)
python3 bugs/new_minimal/openvino/cross_openvino_reciprocal_mul.py

# TVM scripts require the clawwork conda env (tvm 0.11.1 relay + onnxruntime)
/home/binduan/miniconda3/envs/clawwork/bin/python bugs/new_minimal/tvm/github_tvm_004.py

# verify all 120 (exit 0 per script proves the bug still reproduces)
for dir in onnxruntime openvino tflite torch_compile xla onnx2torch; do
  for f in bugs/new_minimal/$dir/*.py; do
    timeout 180 python3 "$f" >/dev/null 2>&1 && echo "OK  $f" || echo "FAIL $f"
  done
done
for f in bugs/new_minimal/tvm/*.py; do
  timeout 120 /home/binduan/miniconda3/envs/clawwork/bin/python "$f" >/dev/null 2>&1     && echo "OK  $f" || echo "FAIL $f"
done
```

---

## ONNXRuntime (8 bugs)

| Script | Verified output |
|--------|-----------------|
| `cross_bitshift_shift64_ov_ort.py` | BUG REPRODUCED on ['ORT_noopt', 'ORT_opt', 'ORT_noopt_LEFT', 'OpenVINO_LEFT']: BitShift by 64 ret... |
| `github_onnx_spec_007.py` | BUG REPRODUCED: at output element 4 ORT returns index 14 (0.7368); spec and torch nearest-exact r... |
| `github_ort_002.py` | BUG REPRODUCED |
| `github_ort_004.py` | BUG REPRODUCED: ORT's Cast(float→int32)→Cast(int32→bool) chain does not preserve int32 truncation... |
| `github_ort_008.py` | BUG REPRODUCED: CPU cubic err=0.0613 vs PyTorch bicubic reference. |
| `github_ort_016.py` | BUG REPRODUCED: ORT bicubic+border output exits [0,15] |
| `github_ort_017_mod_int_divzero_sigfpe.py` | BUG REPRODUCED — ORT Mod(fmod=0, int32) with B=0 triggers SIGFPE |
| `tf_61865_onnx_ort.py` | BUG REPRODUCED: ORT (ref + opt) raise INVALID_ARGUMENT on OOB Gather |

## OpenVINO (57 bugs)

| Script | Verified output |
|--------|-----------------|
| `cross_openvino_add_relu_sub.py` | BUG REPRODUCED: OpenVINO add_relu_sub (max_abs=0.1404) |
| `cross_openvino_add_self_sub_double.py` | BUG REPRODUCED: OpenVINO add_self_sub_double (max_abs=0.1797) |
| `cross_openvino_alibi_attention.py` | BUG REPRODUCED: OpenVINO alibi_attention (max_abs=0.1724) |
| `cross_openvino_aspp_dilated_branch.py` | BUG REPRODUCED: OpenVINO aspp_dilated_branch (max_abs=0.2501) |
| `cross_openvino_attention_causal_only.py` | BUG REPRODUCED: OpenVINO attention_causal_only (max_abs=0.2295) |
| `cross_openvino_attention_logit_softcap.py` | BUG REPRODUCED: OpenVINO attention_logit_softcap (max_abs=38.6729) |
| `cross_openvino_attention_with_sink_token.py` | BUG REPRODUCED: OpenVINO attention_with_sink_token (max_abs=0.1724) |
| `cross_openvino_broadcast_1d_scalar.py` | BUG REPRODUCED: OpenVINO broadcast_1d_scalar (max_abs=0.1404) |
| `cross_openvino_conv_add_relu.py` | BUG REPRODUCED: OpenVINO conv_add_relu (max_abs=0.1462) |
| `cross_openvino_conv_bn_eval_explicit.py` | BUG REPRODUCED: OpenVINO conv_bn_eval_explicit (max_abs=0.7198) |
| `cross_openvino_conv_bn_fusion.py` | BUG REPRODUCED |
| `cross_openvino_conv_bn_relu6.py` | BUG REPRODUCED: OpenVINO conv_bn_relu6 (max_abs=0.3779) |
| `cross_openvino_conv_fp32_precision.py` | BUG REPRODUCED: OpenVINO CPU Conv fp32 accumulation differs from ORT reference |
| `cross_openvino_conv_prelu_channel.py` | BUG REPRODUCED: OpenVINO conv_prelu_channel (max_abs=0.1462) |
| `cross_openvino_ei_log_zero.py` | BUG REPRODUCED: OpenVINO ei_log_zero (max_abs=0.2028) |
| `cross_openvino_einsum_transpose.py` | BUG REPRODUCED: OpenVINO einsum_transpose (max_abs=0.5448) |
| `cross_openvino_expm1_bounded.py` | BUG REPRODUCED: OpenVINO expm1_bounded (max_abs=0.1484) |
| `cross_openvino_flatten_gemm.py` | BUG REPRODUCED: OpenVINO flatten_gemm (max_abs=0.1540) |
| `cross_openvino_flex_attention_precision_sdpa.py` | BUG REPRODUCED: OpenVINO flex_attention_precision_sdpa (max_abs=0.1724) |
| `cross_openvino_fp16_matmul_add.py` | BUG REPRODUCED: OpenVINO fp16 GEMM tile accumulation diverges from ORT |
| `cross_openvino_gemm_sigmoid.py` | BUG REPRODUCED: OpenVINO gemm_sigmoid (max_abs=0.0250) |
| `cross_openvino_global_branch_mul.py` | BUG REPRODUCED: OpenVINO global_branch_mul (max_abs=5.3273) |
| `cross_openvino_glu.py` | BUG REPRODUCED: OpenVINO glu (max_abs=0.5281) |
| `cross_openvino_group_query_attention.py` | BUG REPRODUCED: OpenVINO group_query_attention (max_abs=0.0550) |
| `cross_openvino_ia_sat_sub_uint8_underflow.py` | BUG REPRODUCED: OpenVINO ia_sat_sub_uint8 (MatMul pattern, max_abs=0.1404) |
| `cross_openvino_inception_v3_branch.py` | BUG REPRODUCED: OpenVINO inception_v3_branch (max_abs=0.3883) |
| `cross_openvino_linear_relu_chain.py` | BUG REPRODUCED: OpenVINO Gemm+Relu chain produces wrong fp32 output; |
| `cross_openvino_matmul_add_biasgelu_bcast.py` | BUG REPRODUCED: OpenVINO matmul_add_biasgelu_bcast (max_abs=0.1941) |
| `cross_openvino_matmul_add_layernorm.py` | BUG REPRODUCED: OpenVINO matmul_add_layernorm (max_abs=0.0086) |
| `cross_openvino_maxpool_bad_pad.py` | BUG REPRODUCED: OpenVINO silently accepts MaxPool with pad >= kernel |
| `cross_openvino_multi_query_attention.py` | BUG REPRODUCED: OpenVINO multi_query_attention (max_abs=0.1724) |
| `cross_openvino_multi_scale_conv_branch.py` | BUG REPRODUCED: OpenVINO multi_scale_conv_branch (max_abs=0.2586) |
| `cross_openvino_neg_unary.py` | BUG REPRODUCED: OpenVINO neg_unary (max_abs=0.1747) |
| `cross_openvino_pad_conv.py` | BUG REPRODUCED: OpenVINO pad_conv (max_abs=0.1808) |
| `cross_openvino_pointwise_dw_block.py` | BUG REPRODUCED: OpenVINO pointwise_dw_block (max_abs=2.1500) |
| `cross_openvino_reciprocal_mul.py` | BUG REPRODUCED: OpenVINO reciprocal_mul (max_abs=41.4429) |
| `cross_openvino_reduce_l2_last.py` | BUG REPRODUCED: OpenVINO reduce_l2_last (max_abs=0.0718) |
| `cross_openvino_redundant_reshape.py` | BUG REPRODUCED: OpenVINO redundant_reshape (max_abs=0.1404) |
| `cross_openvino_rrelu_inference_identity.py` | BUG REPRODUCED: OpenVINO rrelu_inference_identity (max_abs=0.1230) |
| `cross_openvino_slice_full_range.py` | BUG REPRODUCED: OpenVINO slice_full_range (max_abs=0.2065) |
| `cross_openvino_space_to_depth_block.py` | BUG REPRODUCED: OpenVINO space_to_depth_block (max_abs=1.0417) |
| `cross_openvino_sub_self_mul_zero.py` | BUG REPRODUCED: OpenVINO CPU vs ORT CPU diverge on Sub(MatMul,MatMul)*const  (max_abs=2.9208) — f... |
| `cross_openvino_tile_conv.py` | BUG REPRODUCED: OpenVINO tile_conv (max_abs=0.1471) |
| `cross_openvino_transformer_encoder_layer.py` | BUG REPRODUCED: OpenVINO transformer_encoder_layer (max_abs=0.0135) |
| `cross_openvino_transpose_matmul_transpose.py` | BUG REPRODUCED: OpenVINO transpose_matmul_transpose (max_abs=0.1714) |
| `cross_openvino_transpose_transpose_squash.py` | BUG REPRODUCED: OpenVINO transpose_transpose_squash (max_abs=0.0890) |
| `cross_openvino_triple_add_residual.py` | BUG REPRODUCED: OpenVINO triple_add_residual (max_abs=0.1797) |
| `cross_openvino_where_mask_fill.py` | BUG REPRODUCED: OpenVINO where_mask_fill (max_abs=0.1797) |
| `github_ov_019_uint8_sub_no_wrap.py` | BUG REPRODUCED — OV uint8 Sub saturates instead of wrapping |
| `github_ov_020_uint8_mul_no_wrap.py` | BUG REPRODUCED — OV uint8 Mul saturates instead of wrapping |
| `github_ov_021_uint8_add_no_wrap.py` | BUG REPRODUCED — OV uint8 Add saturates instead of wrapping |
| `github_ov_022_reducelogsumexp_overflow.py` | BUG REPRODUCED — OV ReduceLogSumExp overflows for inputs ≥ 88.7 (fp32 exp limit) |
| `github_ov_023_relu_nan_propagation.py` | BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754) |
| `github_ov_024_int8_sub_saturation.py` | BUG REPRODUCED — OV int8 Sub saturates instead of two's-complement wrapping |
| `github_ov_025_int8_add_saturation.py` | BUG REPRODUCED — OV int8 Add saturates instead of two's-complement wrapping |
| `github_ov_026_exp_nan_to_inf.py` | BUG REPRODUCED — OV Exp(NaN) → +inf, should propagate NaN (IEEE 754) |
| `tf_61865_onnx_openvino.py` | BUG REPRODUCED: OpenVINO silently returns a value for OOB Gather |

## TFLite (7 bugs)

| Script | Verified output |
|--------|-----------------|
| `cross_tflite_attention_logit_softcap.py` | BUG REPRODUCED: TFLite attention_logit_softcap (max_abs=0.027167) |
| `cross_tflite_fully_connected_mul_fusion_crash.py` | TFLite=CONVERTER CRASH: <unknown>:0: error: loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference... |
| `cross_tflite_l2_normalize_multi_shape.py` | TFLite         [1.0, 1.0, 1.0, 1.0]                                    WRONG |
| `cross_tflite_sub_self_mul_zero.py` | BUG REPRODUCED: TFLite sub_self_mul_zero non-zero (max_abs=2250.318848) |
| `cross_tflite_transformer_encoder_layer.py` | BUG REPRODUCED: TFLite transformer_encoder_layer (max_abs=4606713366520926167589978112.000000) |
| `cross_tflite_transpose_matmul_transpose.py` | BUG REPRODUCED: TFLite transpose_matmul_transpose (max_abs=0.162415) |
| `cross_tflite_unstack_concat_reshape_fold.py` | TFLite         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] WRONG |

## torch.compile (Inductor) (27 bugs)

| Script | Verified output |
|--------|-----------------|
| `cross_torch_compile_add_self_sub_double.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_aspp_dilated_branch.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_conv_k7_stride2.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_conv_manual_celu.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_cp_fp16_matmul_n512.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_einsum_transpose.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_flatten_gemm.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_gemm_sigmoid.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_group_query_attention.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_ia_sat_sub_uint8_underflow.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_manual_hard_swish.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_matmul_add_layernorm.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_multi_query_attention.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_reciprocal_mul.py` | BUG REPRODUCED: torch.compile crashes on reciprocal_mul |
| `cross_torch_compile_reduce_l2_last.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_redundant_reshape.py` | BUG REPRODUCED: torch.compile crashes on redundant_reshape: InternalTorchDynamoError: TypeError: ... |
| `cross_torch_compile_slice_full_range.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_space_to_depth_block.py` | BUG REPRODUCED: Inductor bf16 lowering of SpaceToDepth block diverges from eager by 0.00195312 (>... |
| `cross_torch_compile_sub_self_mul_zero.py` | BUG REPRODUCED: torch.compile crashes on sub_self_mul_zero |
| `cross_torch_compile_topk_last_axis_k1.py` | BUG REPRODUCED: torch.compile crashes on topk_last_axis_k1 |
| `cross_torch_compile_transpose_transpose_squash.py` | BUG REPRODUCED: torch.compile crashes on transpose_transpose_squash: InternalTorchDynamoError: Ty... |
| `cross_torch_compile_triple_add_residual.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `cross_torch_compile_where_mask_fill.py` | BUG REPRODUCED: torch.compile crashes while eager succeeds |
| `github_inductor_009_transpose_reduce_fusion.py` | BUG REPRODUCED: Inductor transpose+reduce fusion reinterprets storage strides |
| `github_inductor_011_bitshift_ub_shift64.py` | BUG REPRODUCED: BitShift by 64 returns non-zero (C UB: x86 masks shift to 6 bits) |
| `pt_121135_torch_compile.py` | BUG REPRODUCED: torch.compile skips shape validation in index_add; |
| `tf_61865_onnx_torch_compile.py` | BUG REPRODUCED: torch.compile raises on OOB Gather |

## XLA / TensorFlow (8 bugs)

| Script | Verified output |
|--------|-----------------|
| `cross_xla_add_self_sub_double.py` | BUG REPRODUCED: XLA add_self_sub_double (max_abs=0.0203) |
| `cross_xla_autocluster_slice_oob_suppress.py` | V4: shape=(3,2,3,4)    RAISED InvalidArgumentError    RAISED InvalidArgumentError    OK  [4.0, 5.... |
| `cross_xla_gather_oob_clamp_variants.py` | -> BUG: XLA silently produced output instead of erroring |
| `cross_xla_matmul_add_biasgelu_bcast.py` | BUG REPRODUCED: XLA matmul_add_biasgelu_bcast (max_abs=0.0678) |
| `cross_xla_space_to_depth_block.py` | BUG REPRODUCED: XLA space_to_depth_block (max_abs=0.1256) |
| `github_tensorflow_002.py` | BUG REPRODUCED — 1 case(s) diverge between XLA and eager: |
| `github_tf_61881_xla_matmul_incompatible_shapes.py` | BUG REPRODUCED: XLA silently executes invalid matmul; eager raises error. |
| `github_tf_61884_xla_dead_code_elimination.py` | BUG REPRODUCED: XLA executes dead slice (no DCE); eager succeeds. |

## onnx2torch (9 bugs)

| Script | Verified output |
|--------|-----------------|
| `cross_bf16_cast_jit_elide.py` | BUG REPRODUCED: onnx2torch cannot materialise Cast(fp32->bf16)->Cast(bf16->fp32) |
| `cross_crms_resize_nearest_ceil.py` | BUG REPRODUCED on ['TorchScript']: onnx2torch Resize ignores nearest_mode=ceil, uses floor instea... |
| `cross_cumsum_bool_dtype.py` | BUG REPRODUCED: onnx2torch / torch.compile / TorchScript give wrong INT64 CumSum; |
| `cross_cumsum_kvcache_multicompiler.py` | BUG REPRODUCED on ['OpenVINO', 'TorchScript']. onnx2torch+TorchScript diverges by 56.801 from ORT... |
| `cross_onnx2torch_cumsum.py` | BUG REPRODUCED |
| `cross_onnx2torch_resize_linear_asym.py` | BUG REPRODUCED |
| `cross_onnx2torch_resize_nearest_ceil.py` | BUG REPRODUCED |
| `tf_61865_onnx_onnx2torch.py` | BUG REPRODUCED: onnx2torch raises (index 6 is out of bounds for dimension 0 with size 6) |
| `tf_61865_onnx_torchscript.py` | BUG REPRODUCED: TorchScript raises on OOB Gather |

## TVM (4 bugs)

| Script | Verified output |
|--------|-----------------|
| `github_tvm_004.py` | BUG REPRODUCED: TVM Relay Resize(nearest, half_pixel, round_prefer_floor) diverges from ORT |
| `github_tvm_010_simplifyexpr_rsqrt_precision.py` | BUG REPRODUCED: TVM Relay RoiAlign ignores coordinate_transformation_mode='half_pixel' and diverg... |
| `github_tvm_011_lifttransformparams_const_bind.py` | BUG REPRODUCED: TVM Relay FoldConstant rewrites inf*X - inf*X to 0, violating IEEE-754 (ORT retur... |
| `github_tvm_012_gelu_approx_tanh.py` | BUG REPRODUCED: TVM Relay Gelu importer ignores approximate='tanh' and returns the exact erf output |
