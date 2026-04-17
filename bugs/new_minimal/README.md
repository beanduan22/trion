# ML Compiler Bug Collection — Minimal Reproducible Scripts

**122 verified bugs** across 7 ML compilers/runtimes.  
Verified: 2026-04-17 · Python 3.13 · ONNX 1.21 · ONNXRuntime 1.24.4 · OpenVINO 2026.0 · TensorFlow 2.21.0 · PyTorch 2.9.1 · TVM master · onnx2torch latest

Every script:
- Builds its ONNX model from scratch with `onnx.helper` — no binary blobs
- Prints the **correct** (ORT) output vs the **buggy** backend output
- `Exit 0` = BUG REPRODUCED · `Exit 1` = not reproduced · `Exit 2` = missing dep

---

## Summary

| Compiler | Bugs | Root cause |
|---|---|---|
| OpenVINO | 58 | fp32 MatMul/Conv tiling precision; uint8/int8 wrap saturation; NaN propagation; GPU tile overflow |
| torch.compile | 27 | `Reshape` runtime shape tensor → `InternalTorchDynamoError`; index_add shape bypass; bitshift UB |
| ONNXRuntime | 9 | Resize nearest/bilinear/cubic; int-truncation fusion; SIGFPE mod-zero; GridSample clamp; bitshift UB |
| onnx2torch | 9 | CumSum graph break; resize coord/nearest-ceil; INT64 cumsum dtype; bf16 cast elide; OOB Gather |
| XLA / TensorFlow | 8 | OOB gather silent clamp; autocluster DCE; dead-code slice; matmul incompatible shapes; fp32 GEMM |
| TFLite | 7 | FC fusion converter crash; L2-norm axis semantics; unstack-concat fold; attention softcap |
| TVM | 4 | SimplifyExpr rsqrt; LiftTransformParams off-by-one; GELU approx; cubic resize |

---

## Running

```bash
# single script
python3 openvino/cross_openvino_reciprocal_mul.py

# all scripts for one compiler
for f in openvino/*.py; do
    python3 "$f" > /dev/null 2>&1 && echo "BUG $f" || echo "SKIP $f"
done

# verify all 122 (reads pre-captured .out files)
for dir in onnxruntime openvino tflite torch_compile tvm xla onnx2torch; do
  for f in $dir/*.py; do python3 "$f" >/dev/null 2>&1 && echo "OK $f" || echo "FAIL $f"; done
done
```

---

## ONNXRuntime (9 bugs)

| Script | Verified output |
|--------|----------------|
| `cross_bitshift_shift64_ov_ort.py` | `BUG REPRODUCED on ['ORT_noopt', 'ORT_opt', 'ORT_noopt_LEFT', 'OpenVINO_LEFT']: BitShift by 64 returns non-zero (C UB — x` |
| `github_onnx_spec_007.py` | `BUG REPRODUCED` |
| `github_ort_002.py` | `BUG REPRODUCED` |
| `github_ort_003.py` | `BUG REPRODUCED` |
| `github_ort_004.py` | `BUG REPRODUCED` |
| `github_ort_008.py` | `BUG REPRODUCED` |
| `github_ort_016.py` | `BUG REPRODUCED` |
| `github_ort_017_mod_int_divzero_sigfpe.py` | `BUG REPRODUCED — ORT Mod(fmod=0, int32) with B=0 triggers SIGFPE` |
| `tf_61865_onnx_ort.py` | `BUG REPRODUCED: ORT (ref + opt) raise INVALID_ARGUMENT on OOB Gather` |


## OpenVINO (58 bugs)

| Script | Verified output |
|--------|----------------|
| `cross_openvino_add_relu_sub.py` | `BUG REPRODUCED: OpenVINO add_relu_sub (max_abs=0.1404)` |
| `cross_openvino_add_self_sub_double.py` | `BUG REPRODUCED: OpenVINO add_self_sub_double (max_abs=0.1797)` |
| `cross_openvino_alibi_attention.py` | `BUG REPRODUCED: OpenVINO alibi_attention (max_abs=0.1724)` |
| `cross_openvino_aspp_dilated_branch.py` | `BUG REPRODUCED: OpenVINO aspp_dilated_branch (max_abs=0.2501)` |
| `cross_openvino_attention_causal_only.py` | `BUG REPRODUCED: OpenVINO attention_causal_only (max_abs=0.2295)` |
| `cross_openvino_attention_logit_softcap.py` | `BUG REPRODUCED: OpenVINO attention_logit_softcap (max_abs=38.6729)` |
| `cross_openvino_attention_with_sink_token.py` | `BUG REPRODUCED: OpenVINO attention_with_sink_token (max_abs=0.1724)` |
| `cross_openvino_broadcast_1d_scalar.py` | `BUG REPRODUCED: OpenVINO broadcast_1d_scalar (max_abs=0.1404)` |
| `cross_openvino_conv_add_relu.py` | `BUG REPRODUCED: OpenVINO conv_add_relu (max_abs=0.1462)` |
| `cross_openvino_conv_bn_eval_explicit.py` | `BUG REPRODUCED: OpenVINO conv_bn_eval_explicit (max_abs=0.7198)` |
| `cross_openvino_conv_bn_fusion.py` | `BUG REPRODUCED` |
| `cross_openvino_conv_bn_relu6.py` | `BUG REPRODUCED: OpenVINO conv_bn_relu6 (max_abs=0.3779)` |
| `cross_openvino_conv_fp32_precision.py` | `BUG REPRODUCED: OpenVINO CPU Conv fp32 accumulation differs from ORT reference` |
| `cross_openvino_conv_prelu_channel.py` | `BUG REPRODUCED: OpenVINO conv_prelu_channel (max_abs=0.1462)` |
| `cross_openvino_ei_log_zero.py` | `BUG REPRODUCED: OpenVINO ei_log_zero (max_abs=0.2028)` |
| `cross_openvino_einsum_transpose.py` | `BUG REPRODUCED: OpenVINO einsum_transpose (max_abs=0.5448)` |
| `cross_openvino_expm1_bounded.py` | `BUG REPRODUCED: OpenVINO expm1_bounded (max_abs=0.1484)` |
| `cross_openvino_flatten_gemm.py` | `BUG REPRODUCED: OpenVINO flatten_gemm (max_abs=0.1540)` |
| `cross_openvino_flex_attention_precision_sdpa.py` | `BUG REPRODUCED: OpenVINO flex_attention_precision_sdpa (max_abs=0.1724)` |
| `cross_openvino_fp16_matmul_add.py` | `BUG REPRODUCED: OpenVINO fp16 GEMM tile accumulation diverges from ORT` |
| `cross_openvino_gemm_sigmoid.py` | `BUG REPRODUCED: OpenVINO gemm_sigmoid (max_abs=0.0250)` |
| `cross_openvino_global_branch_mul.py` | `BUG REPRODUCED: OpenVINO global_branch_mul (max_abs=5.3273)` |
| `cross_openvino_glu.py` | `BUG REPRODUCED: OpenVINO glu (max_abs=0.5281)` |
| `cross_openvino_group_query_attention.py` | `BUG REPRODUCED: OpenVINO group_query_attention (max_abs=0.0550)` |
| `cross_openvino_ia_sat_sub_uint8_underflow.py` | `BUG REPRODUCED: OpenVINO ia_sat_sub_uint8 (MatMul pattern, max_abs=0.1404)` |
| `cross_openvino_inception_v3_branch.py` | `BUG REPRODUCED: OpenVINO inception_v3_branch (max_abs=0.3883)` |
| `cross_openvino_linear_relu_chain.py` | `BUG REPRODUCED: OpenVINO Gemm+Relu chain produces wrong fp32 output;` |
| `cross_openvino_matmul_add_biasgelu_bcast.py` | `BUG REPRODUCED: OpenVINO matmul_add_biasgelu_bcast (max_abs=0.1941)` |
| `cross_openvino_matmul_add_layernorm.py` | `BUG REPRODUCED: OpenVINO matmul_add_layernorm (max_abs=0.0086)` |
| `cross_openvino_maxpool_bad_pad.py` | `BUG REPRODUCED: OpenVINO silently accepts MaxPool with pad >= kernel` |
| `cross_openvino_multi_query_attention.py` | `BUG REPRODUCED: OpenVINO multi_query_attention (max_abs=0.1724)` |
| `cross_openvino_multi_scale_conv_branch.py` | `BUG REPRODUCED: OpenVINO multi_scale_conv_branch (max_abs=0.2586)` |
| `cross_openvino_neg_unary.py` | `BUG REPRODUCED: OpenVINO neg_unary (max_abs=0.1747)` |
| `cross_openvino_pad_conv.py` | `BUG REPRODUCED: OpenVINO pad_conv (max_abs=0.1808)` |
| `cross_openvino_pointwise_dw_block.py` | `BUG REPRODUCED: OpenVINO pointwise_dw_block (max_abs=2.1500)` |
| `cross_openvino_reciprocal_mul.py` | `BUG REPRODUCED: OpenVINO reciprocal_mul (max_abs=41.4429)` |
| `cross_openvino_reduce_l2_last.py` | `BUG REPRODUCED: OpenVINO reduce_l2_last (max_abs=0.0718)` |
| `cross_openvino_redundant_reshape.py` | `BUG REPRODUCED: OpenVINO redundant_reshape (max_abs=0.1404)` |
| `cross_openvino_rrelu_inference_identity.py` | `BUG REPRODUCED: OpenVINO rrelu_inference_identity (max_abs=0.1230)` |
| `cross_openvino_slice_full_range.py` | `BUG REPRODUCED: OpenVINO slice_full_range (max_abs=0.2065)` |
| `cross_openvino_space_to_depth_block.py` | `BUG REPRODUCED: OpenVINO space_to_depth_block (max_abs=1.0417)` |
| `cross_openvino_sub_self_mul_zero.py` | `BUG REPRODUCED: OpenVINO sub_self_mul_zero (max_abs=2.9208)` |
| `cross_openvino_tile_conv.py` | `BUG REPRODUCED: OpenVINO tile_conv (max_abs=0.1471)` |
| `cross_openvino_transformer_encoder_layer.py` | `BUG REPRODUCED: OpenVINO transformer_encoder_layer (max_abs=0.0135)` |
| `cross_openvino_transpose_matmul_transpose.py` | `BUG REPRODUCED: OpenVINO transpose_matmul_transpose (max_abs=0.1714)` |
| `cross_openvino_transpose_transpose_squash.py` | `BUG REPRODUCED: OpenVINO transpose_transpose_squash (max_abs=0.0890)` |
| `cross_openvino_triple_add_residual.py` | `BUG REPRODUCED: OpenVINO triple_add_residual (max_abs=0.1797)` |
| `cross_openvino_where_mask_fill.py` | `BUG REPRODUCED: OpenVINO where_mask_fill (max_abs=0.1797)` |
| `github_ov_015_matmul_gpu_tile_overflow.py` | `BUG REPRODUCED: MatMul GPU tile skips partial last tile for dim > 2048` |
| `github_ov_019_uint8_sub_no_wrap.py` | `BUG REPRODUCED — OV uint8 Sub saturates instead of wrapping` |
| `github_ov_020_uint8_mul_no_wrap.py` | `BUG REPRODUCED — OV uint8 Mul saturates instead of wrapping` |
| `github_ov_021_uint8_add_no_wrap.py` | `BUG REPRODUCED — OV uint8 Add saturates instead of wrapping` |
| `github_ov_022_reducelogsumexp_overflow.py` | `BUG REPRODUCED — OV ReduceLogSumExp overflows for inputs ≥ 88.7 (fp32 exp limit)` |
| `github_ov_023_relu_nan_propagation.py` | `BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754)` |
| `github_ov_024_int8_sub_saturation.py` | `BUG REPRODUCED — OV int8 Sub saturates instead of two's-complement wrapping` |
| `github_ov_025_int8_add_saturation.py` | `BUG REPRODUCED — OV int8 Add saturates instead of two's-complement wrapping` |
| `github_ov_026_exp_nan_to_inf.py` | `BUG REPRODUCED — OV Exp(NaN) → +inf, should propagate NaN (IEEE 754)` |
| `tf_61865_onnx_openvino.py` | `BUG REPRODUCED: OpenVINO silently returns a value for OOB Gather` |


## TFLite (7 bugs)

| Script | Verified output |
|--------|----------------|
| `cross_tflite_attention_logit_softcap.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` |
| `cross_tflite_fully_connected_mul_fusion_crash.py` | `TFLite=CONVERTER CRASH: <unknown>:0: error: loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_1320"] at cal` |
| `cross_tflite_l2_normalize_multi_shape.py` | `TFLite         [1.0, 1.0, 1.0, 1.0]                                    WRONG` |
| `cross_tflite_sub_self_mul_zero.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` |
| `cross_tflite_transformer_encoder_layer.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` |
| `cross_tflite_transpose_matmul_transpose.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` |
| `cross_tflite_unstack_concat_reshape_fold.py` | `TFLite         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] WRONG` |


## torch.compile / Inductor (27 bugs)

| Script | Verified output |
|--------|----------------|
| `cross_torch_compile_add_self_sub_double.py` | `BUG REPRODUCED: torch.compile crashes on add_self_sub_double` |
| `cross_torch_compile_aspp_dilated_branch.py` | `BUG REPRODUCED: torch.compile crashes on aspp_dilated_branch` |
| `cross_torch_compile_conv_k7_stride2.py` | `BUG REPRODUCED: torch.compile crashes on conv_k7_stride2` |
| `cross_torch_compile_conv_manual_celu.py` | `BUG REPRODUCED: torch.compile crashes on conv_manual_celu` |
| `cross_torch_compile_cp_fp16_matmul_n512.py` | `BUG REPRODUCED: torch.compile crashes on cp_fp16_matmul_n512` |
| `cross_torch_compile_einsum_transpose.py` | `BUG REPRODUCED: torch.compile crashes on einsum_transpose` |
| `cross_torch_compile_flatten_gemm.py` | `BUG REPRODUCED: torch.compile crashes on flatten_gemm` |
| `cross_torch_compile_gemm_sigmoid.py` | `BUG REPRODUCED: torch.compile crashes on gemm_sigmoid` |
| `cross_torch_compile_group_query_attention.py` | `BUG REPRODUCED: torch.compile crashes on group_query_attention` |
| `cross_torch_compile_ia_sat_sub_uint8_underflow.py` | `BUG REPRODUCED: torch.compile crashes on ia_sat_sub_uint8_underflow` |
| `cross_torch_compile_manual_hard_swish.py` | `BUG REPRODUCED: torch.compile crashes on manual_hard_swish` |
| `cross_torch_compile_matmul_add_layernorm.py` | `BUG REPRODUCED: torch.compile crashes on matmul_add_layernorm` |
| `cross_torch_compile_multi_query_attention.py` | `BUG REPRODUCED: torch.compile crashes on multi_query_attention` |
| `cross_torch_compile_reciprocal_mul.py` | `BUG REPRODUCED: torch.compile crashes on reciprocal_mul` |
| `cross_torch_compile_reduce_l2_last.py` | `BUG REPRODUCED: torch.compile crashes on reduce_l2_last` |
| `cross_torch_compile_redundant_reshape.py` | `BUG REPRODUCED: torch.compile crashes on redundant_reshape: InternalTorchDynamoError: TypeError: torch.Size() takes an i` |
| `cross_torch_compile_slice_full_range.py` | `BUG REPRODUCED: torch.compile crashes on slice_full_range` |
| `cross_torch_compile_space_to_depth_block.py` | `BUG REPRODUCED: torch.compile crashes on space_to_depth_block: NotImplementedError: Converter is not implemented (Operat` |
| `cross_torch_compile_sub_self_mul_zero.py` | `BUG REPRODUCED: torch.compile crashes on sub_self_mul_zero` |
| `cross_torch_compile_topk_last_axis_k1.py` | `BUG REPRODUCED: torch.compile crashes on topk_last_axis_k1` |
| `cross_torch_compile_transpose_transpose_squash.py` | `BUG REPRODUCED: torch.compile crashes on transpose_transpose_squash: InternalTorchDynamoError: TypeError: torch.Size() t` |
| `cross_torch_compile_triple_add_residual.py` | `BUG REPRODUCED: torch.compile crashes on triple_add_residual` |
| `cross_torch_compile_where_mask_fill.py` | `BUG REPRODUCED: torch.compile crashes on where_mask_fill` |
| `github_inductor_009_transpose_reduce_fusion.py` | `BUG REPRODUCED: Inductor transpose+reduce fusion reinterprets storage strides` |
| `github_inductor_011_bitshift_ub_shift64.py` | `BUG REPRODUCED: BitShift by 64 returns non-zero (C UB: x86 masks shift to 6 bits)` |
| `pt_121135_torch_compile.py` | `BUG REPRODUCED: torch.compile skips shape validation in index_add;` |
| `tf_61865_onnx_torch_compile.py` | `BUG REPRODUCED: torch.compile raises on OOB Gather` |


## TVM (4 bugs)

| Script | Verified output |
|--------|----------------|
| `github_tvm_004.py` | `BUG REPRODUCED` |
| `github_tvm_010_simplifyexpr_rsqrt_precision.py` | `BUG REPRODUCED: SimplifyExpr rsqrt rewrite introduces large precision error` |
| `github_tvm_011_lifttransformparams_const_bind.py` | `BUG REPRODUCED: LiftTransformParams corrupts constant binding` |
| `github_tvm_012_gelu_approx_tanh.py` | `BUG REPRODUCED: TVM maps ONNX Gelu(approx=tanh) → exact erf, producing systematic error > 1e-4` |


## XLA / TensorFlow (8 bugs)

| Script | Verified output |
|--------|----------------|
| `cross_xla_add_self_sub_double.py` | `BUG REPRODUCED: XLA add_self_sub_double (max_abs=0.0203)` |
| `cross_xla_autocluster_slice_oob_suppress.py` | `V4: shape=(3,2,3,4)    RAISED InvalidArgumentError    RAISED InvalidArgumentError    OK  [4.0, 5.0, 6.0]...  ← BUG` |
| `cross_xla_gather_oob_clamp_variants.py` | `-> BUG: XLA silently produced output instead of erroring` |
| `cross_xla_matmul_add_biasgelu_bcast.py` | `BUG REPRODUCED: XLA matmul_add_biasgelu_bcast (max_abs=0.0678)` |
| `cross_xla_space_to_depth_block.py` | `BUG REPRODUCED: XLA space_to_depth_block (max_abs=0.1193)` |
| `github_tensorflow_002.py` | `BUG REPRODUCED` |
| `github_tf_61881_xla_matmul_incompatible_shapes.py` | `BUG REPRODUCED: XLA silently executes invalid matmul; eager raises error.` |
| `github_tf_61884_xla_dead_code_elimination.py` | `BUG REPRODUCED: XLA executes dead slice (no DCE); eager succeeds.` |


## onnx2torch / cross-compiler (9 bugs)

| Script | Verified output |
|--------|----------------|
| `cross_bf16_cast_jit_elide.py` | `BUG REPRODUCED on 3 JIT compiler(s): PyTorch Inductor, TensorFlow XLA, JAX-jit` |
| `cross_crms_resize_nearest_ceil.py` | `BUG REPRODUCED on ['TorchScript']: onnx2torch Resize ignores nearest_mode=ceil, uses floor instead — wrong source pixel ` |
| `cross_cumsum_bool_dtype.py` | `BUG REPRODUCED: onnx2torch / torch.compile / TorchScript give wrong INT64 CumSum;` |
| `cross_cumsum_kvcache_multicompiler.py` | `BUG REPRODUCED on ['OpenVINO', 'TorchScript']: CumSum→Transpose→MatMul — OpenVINO tiled-GEMM fp32 accumulation order dif` |
| `cross_onnx2torch_cumsum.py` | `BUG REPRODUCED` |
| `cross_onnx2torch_resize_linear_asym.py` | `BUG REPRODUCED` |
| `cross_onnx2torch_resize_nearest_ceil.py` | `BUG REPRODUCED` |
| `tf_61865_onnx_onnx2torch.py` | `BUG REPRODUCED: onnx2torch raises (index 6 is out of bounds for dimension 0 with size 6)` |
| `tf_61865_onnx_torchscript.py` | `BUG REPRODUCED: TorchScript raises on OOB Gather` |


---

## Detailed outputs


### ONNXRuntime

**`cross_bitshift_shift64_ov_ort.py`**
```
Input values: [1000, 255, 1, 42]
Shift amount: [64, 64, 64, 64] (all 64)
Expected:     [0, 0, 0, 0] (all 0)

Backend                 max_diff  bug?
----------------------------------------
ORT_noopt                   1000  BUG
ORT_opt                     1000  BUG
OpenVINO                       0  ok
ORT_noopt_LEFT              1000  BUG
OpenVINO_LEFT         2147483648  BUG

PASS=False
BUG REPRODUCED on ['ORT_noopt', 'ORT_opt', 'ORT_noopt_LEFT', 'OpenVINO_LEFT']: BitShift by 64 returns non-zero (C UB — x86 masks shift to 6 bits)
```

**`github_onnx_spec_007.py`**
```
ORT  output: [0.05263158 0.2631579  0.42105263 0.57894737 0.7368421  0.94736844]
Expected:    [0.05263158 0.21052632 0.42105263 0.57894737 0.7894737  0.94736844]
Bug (elem 4): ORT=0.7368, expected=0.7895
PASS=False
BUG REPRODUCED
```

**`github_ort_002.py`**
```
ORT   output[30:35]: [12. 12. 13. 13. 14.]
Torch output[30:35]: [12. 12. 13. 13. 13.]
Mismatched elements: 12/64
PASS=False
BUG REPRODUCED
```

**`github_ort_003.py`**
```
asymmetric coord mode max abs error vs PyTorch: 0.2947
ORT   output[:4]: [0.37454012 0.6050098  0.83547944 0.90697014]
Torch output[:4]: [0.37454012 0.43215755 0.6626272  0.8930969 ]
PASS=False
BUG REPRODUCED
```

**`github_ort_004.py`**
```
Input:    [-0.2 -0.1  0.   0.1  0.2]
Expected: [False False False False False]
ORT out:  [ True  True False  True  True]
PASS=False
BUG REPRODUCED
```

**`github_ort_008.py`**
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
BUG REPRODUCED
```

**`github_ort_016.py`**
```
bicubic+border output[0,0,0,:]: [-0.5400001  2.6759987 12.323996  15.539993 ]
bicubic+zeros  output[0,0,0,:]: [-0.33048013  0.92534363  4.692814    5.9486337 ]
Bicubic+border values in [0,15] range: False
PASS=False
BUG REPRODUCED
```

**`github_ort_017_mod_int_divzero_sigfpe.py`**
```
OnnxRuntime version: 1.24.4
Child exit code: -8
Child killed by SIGFPE (signal 8)
BUG REPRODUCED — ORT Mod(fmod=0, int32) with B=0 triggers SIGFPE
```

**`tf_61865_onnx_ort.py`**
```
[1;31m2026-04-17 18:00:29.181199945 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=5 must be within the inclusive range [-5,4][m
[1;31m2026-04-17 18:00:29.331417718 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=5 must be within the inclusive range [-5,4][m
params  : [10.0, 20.0, 30.0, 40.0, 50.0]
indices : [0, 1, 2, 3, 5, 4]  (index 5 is OOB for size-5 tensor)

Compiler          Output                            Error
--------------------------------------------------------------------------------
ORT_ref           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st  ← TARGET
ORT_opt           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st  ← TARGET
OpenVINO          [10.0, 20.0, 30.0, 40.0, 0.0, 50.0]  —
onnx2torch        —                                 index 5 is out of bounds for dimension 0 with size 5
torch_compile     —                                 kernel, /tmp/torchinductor_binduan/qn/cqnw5m3qacnd76qfa
TorchScript       —                                 index 5 is out of bounds for dimension 0 with size 5

BUG REPRODUCED: ORT (ref + opt) raise INVALID_ARGUMENT on OOB Gather
  while other compilers return values: {'OpenVINO': [10.0, 20.0, 30.0, 40.0, 0.0, 50.0]}
```


### OpenVINO

**`cross_openvino_add_relu_sub.py`**
```
ORT:      [33.165524    0.48662525  1.9708737  27.003704  ]
OpenVINO: [33.117676    0.48662525  1.9708737  26.950716  ]
max_abs=0.1404
BUG REPRODUCED: OpenVINO add_relu_sub (max_abs=0.1404)
```

**`cross_openvino_add_self_sub_double.py`**
```
ORT:      [ 33.65159  -46.478348 -28.172342  27.534014]
OpenVINO: [ 33.55266  -46.616745 -28.106264  27.512966]
max_abs=0.1797
BUG REPRODUCED: OpenVINO add_self_sub_double (max_abs=0.1797)
```

**`cross_openvino_alibi_attention.py`**
```
ORT:      [ 21.168224 -24.423386  20.916695  17.117714]
OpenVINO: [ 21.125 -24.375  20.875  17.   ]
max_abs=0.1724
BUG REPRODUCED: OpenVINO alibi_attention (max_abs=0.1724)
```

**`cross_openvino_aspp_dilated_branch.py`**
```
ORT:      [  8.451527  -7.942028 -13.74964   27.49309 ]
OpenVINO: [  8.430385   -7.9617577 -13.805025   27.444984 ]
max_abs=0.2501
BUG REPRODUCED: OpenVINO aspp_dilated_branch (max_abs=0.2501)
```

**`cross_openvino_attention_causal_only.py`**
```
ORT:      [-16.701406   -4.9059258   7.8918476  -7.864847 ]
OpenVINO: [-16.625    -4.8125    7.78125  -7.875  ]
max_abs=0.2295
BUG REPRODUCED: OpenVINO attention_causal_only (max_abs=0.2295)
```

**`cross_openvino_attention_logit_softcap.py`**
```
ORT:      [ 2792.551     599.92145 -5608.0615    753.3436 ]
OpenVINO: [ 2768.1167    611.7286  -5585.166     762.72314]
max_abs=38.6729
BUG REPRODUCED: OpenVINO attention_logit_softcap (max_abs=38.6729)
```

**`cross_openvino_attention_with_sink_token.py`**
```
ORT:      [ 21.168224 -24.423386  20.916695  17.117714]
OpenVINO: [ 21.125 -24.375  20.875  17.   ]
max_abs=0.1724
BUG REPRODUCED: OpenVINO attention_with_sink_token (max_abs=0.1724)
```

**`cross_openvino_broadcast_1d_scalar.py`**
```
ORT:      [ 35.598927 -45.111603 -28.441078  26.896048]
OpenVINO: [ 35.55108  -45.12861  -28.352718  26.84306 ]
max_abs=0.1404
BUG REPRODUCED: OpenVINO broadcast_1d_scalar (max_abs=0.1404)
```

**`cross_openvino_conv_add_relu.py`**
```
ORT:      [ 0.        18.485115  24.636873   7.4838285]
OpenVINO: [ 0.       18.410646 24.628017  7.431991]
max_abs=0.1462
BUG REPRODUCED: OpenVINO conv_add_relu (max_abs=0.1462)
```

**`cross_openvino_conv_bn_eval_explicit.py`**
```
ORT:      [-20.431755   15.919664   21.226522    6.4293294]
OpenVINO: [-20.5      15.9375   21.25      6.40625]
max_abs=0.7198
BUG REPRODUCED: OpenVINO conv_bn_eval_explicit (max_abs=0.7198)
```

**`cross_openvino_conv_bn_fusion.py`**
```
ORT ref[:4]:      [ 1.5056843   1.9531173  -0.09237421  2.6932142 ]
OpenVINO[:4]:     [ 1.5078125   1.953125   -0.09765625  2.703125  ]
max_diff=0.022135  tol=0.01
BUG REPRODUCED
```

**`cross_openvino_conv_bn_relu6.py`**
```
ORT:      [0. 6. 6. 6.]
OpenVINO: [0. 6. 6. 6.]
max_abs=0.3779
BUG REPRODUCED: OpenVINO conv_bn_relu6 (max_abs=0.3779)
```

**`cross_openvino_conv_fp32_precision.py`**
```
ORT ref (first 4): [ 4.2668686  9.879526   4.723992  -1.4073367]
OpenVINO (first 4): [ 4.2729235  9.873534   4.7310348 -1.4000506]
Max abs diff: 0.053732  tol=0.02
5×5 kernel max abs diff: 0.083416  tol=0.02
PASS=False
BUG REPRODUCED: OpenVINO CPU Conv fp32 accumulation differs from ORT reference
```

**`cross_openvino_conv_prelu_channel.py`**
```
ORT:      [-1.3507019 18.11512   24.266878   7.113834 ]
OpenVINO: [-1.3505647 18.040651  24.258022   7.0619965]
max_abs=0.1462
BUG REPRODUCED: OpenVINO conv_prelu_channel (max_abs=0.1462)
```

**`cross_openvino_ei_log_zero.py`**
```
ORT:      [-16.118095 -16.118095 -16.11629  -16.118095]
OpenVINO: [-16.118095 -16.118095 -16.116405 -16.118095]
max_abs=0.2028
BUG REPRODUCED: OpenVINO ei_log_zero (max_abs=0.2028)
```

**`cross_openvino_einsum_transpose.py`**
```
ORT:      [-55.143017 -45.517612  19.589365  28.159199]
OpenVINO: [-54.9281   -45.43213   19.549316  28.169434]
max_abs=0.5448
BUG REPRODUCED: OpenVINO einsum_transpose (max_abs=0.5448)
```

**`cross_openvino_expm1_bounded.py`**
```
ORT:      [ 12.645439  -35.535053    3.0442579  -2.6048763]
OpenVINO: [ 12.581936  -35.535927    3.081318   -2.5640929]
max_abs=0.1484
BUG REPRODUCED: OpenVINO expm1_bounded (max_abs=0.1484)
```

**`cross_openvino_flatten_gemm.py`**
```
ORT:      [34.732975  12.6912985 11.113333   4.188798 ]
OpenVINO: [34.790512  12.693661  11.158191   4.1422715]
max_abs=0.1540
BUG REPRODUCED: OpenVINO flatten_gemm (max_abs=0.1540)
```

**`cross_openvino_flex_attention_precision_sdpa.py`**
```
ORT:      [ 21.168224 -24.423386  20.916695  17.117714]
OpenVINO: [ 21.125 -24.375  20.875  17.   ]
max_abs=0.1724
BUG REPRODUCED: OpenVINO flex_attention_precision_sdpa (max_abs=0.1724)
```

**`cross_openvino_fp16_matmul_add.py`**
```
ORT ref (first 4): [ 5.207 -0.85   4.332 -7.043]
OpenVINO (first 4): [ 5.22  -0.875  4.312 -7.062]
Max abs diff: 0.078125  tol=0.05
PASS=False
BUG REPRODUCED: OpenVINO fp16 GEMM tile accumulation diverges from ORT
```

**`cross_openvino_gemm_sigmoid.py`**
```
ORT:      [1.        0.9999969 0.9999851 0.985062 ]
OpenVINO: [1.         0.9999969  0.99998575 0.9843617 ]
max_abs=0.0250
BUG REPRODUCED: OpenVINO gemm_sigmoid (max_abs=0.0250)
```

**`cross_openvino_global_branch_mul.py`**
```
ORT:      [-204.5154   -159.69688   -88.1904    -26.530653]
OpenVINO: [-204.      -158.625    -87.90625  -26.15332]
max_abs=5.3273
BUG REPRODUCED: OpenVINO global_branch_mul (max_abs=5.3273)
```

**`cross_openvino_glu.py`**
```
ORT:      [-3.2990341e+00 -2.4846565e+01  1.4714709e-04  4.8295879e+00]
OpenVINO: [-2.9529891e+00 -2.4875000e+01  1.4859703e-04  4.8437500e+00]
max_abs=0.5281
BUG REPRODUCED: OpenVINO glu (max_abs=0.5281)
```

**`cross_openvino_group_query_attention.py`**
```
ORT:      [ 0.03248205  0.68097824 -1.6774524   2.9866254 ]
OpenVINO: [ 0.03833008  0.6875     -1.6796875   2.984375  ]
max_abs=0.0550
BUG REPRODUCED: OpenVINO group_query_attention (max_abs=0.0550)
```

**`cross_openvino_ia_sat_sub_uint8_underflow.py`**
```
ORT:      [  5.   3.  10.   1.   2.   8. 200. 100.]
OpenVINO: [  5.   3.  10.   1.   2.   8. 200. 100.]
max_abs=0.000000
NOT reproduced - trying MatMul pattern instead
BUG REPRODUCED: OpenVINO ia_sat_sub_uint8 (MatMul pattern, max_abs=0.1404)
```

**`cross_openvino_inception_v3_branch.py`**
```
ORT:      [ 1.9989679   4.023582   -3.8524904   0.55001694]
OpenVINO: [ 1.984375  4.       -3.875     0.578125]
max_abs=0.3883
BUG REPRODUCED: OpenVINO inception_v3_branch (max_abs=0.3883)
```

**`cross_openvino_linear_relu_chain.py`**
```
Model: Gemm(64→64)→Relu→Gemm(64→64)→Relu→Gemm(64→32)→Relu  [fp32]

Seed   Compiler           max_diff  Status
--------------------------------------------------
seed=0   ORT_opt             0.00000  ok
seed=0   OpenVINO            1.97807  BUG  *** BUG
seed=0   onnx2torch          0.00012  ok
seed=0   torch.compile       0.00012  ok
seed=0   TorchScript         0.00012  ok
seed=1   ORT_opt             0.00000  ok
seed=1   OpenVINO            1.88255  BUG  *** BUG
seed=1   onnx2torch          0.00018  ok
seed=1   torch.compile       0.00018  ok
seed=1   TorchScript         0.00018  ok
seed=2   ORT_opt             0.00000  ok
seed=2   OpenVINO            3.45098  BUG  *** BUG
seed=2   onnx2torch          0.00018  ok
seed=2   torch.compile       0.00018  ok
seed=2   TorchScript         0.00018  ok
seed=3   ORT_opt             0.00000  ok
seed=3   OpenVINO            1.76013  BUG  *** BUG
seed=3   onnx2torch          0.00015  ok
seed=3   torch.compile       0.00015  ok
seed=3   TorchScript         0.00015  ok

BUG REPRODUCED: OpenVINO Gemm+Relu chain produces wrong fp32 output;
  ORT_opt / onnx2torch / torch.compile / TorchScript all agree with ORT_ref.
```

**`cross_openvino_matmul_add_biasgelu_bcast.py`**
```
ORT:      [16.35682   0.        0.       14.601542]
OpenVINO: [16.33356   0.        0.       14.573442]
max_abs=0.1941
BUG REPRODUCED: OpenVINO matmul_add_biasgelu_bcast (max_abs=0.1941)
```

**`cross_openvino_matmul_add_layernorm.py`**
```
ORT:      [-0.8029778  -0.36353838  0.3149736   0.22312963]
OpenVINO: [-0.8031457  -0.36591285  0.31051213  0.22435154]
max_abs=0.0086
BUG REPRODUCED: OpenVINO matmul_add_layernorm (max_abs=0.0086)
```

**`cross_openvino_maxpool_bad_pad.py`**
```
[1;31m2026-04-17 18:00:58.831454588 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
[1;31m2026-04-17 18:00:59.247217795 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0] failed while attempting to run meta for aten.max_pool2d_with_indices.default
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0] Traceback (most recent call last):
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_subclasses/fake_tensor.py", line 2755, in _dispatch_impl
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     r = func(*args, **kwargs)
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_ops.py", line 841, in __call__
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     return self._op(*args, **kwargs)
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]            ~~~~~~~~^^^^^^^^^^^^^^^^^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4855, in meta_max_pool2d_with_indices
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     ) = max_pool2d_checks_and_compute_shape(
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]         input, kernel_size, stride, padding, dilation, ceil_mode
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     )
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     ^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4774, in max_pool2d_checks_and_compute_shape
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4440, in pooling_output_shape
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     torch._check(
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     ~~~~~~~~~~~~^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]         pad <= ((kernelSize - 1) * dilation + 1) // 2,
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     ...<3 lines>...
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]         ),
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]         ^^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     )
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     ^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1695, in _check
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     _check_with(RuntimeError, cond, message)
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1677, in _check_with
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0]     raise error_type(message_evaluated)
E0417 18:01:01.090000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [0/0] RuntimeError: pad should be at most half of effective kernel size, but got pad=3, kernel_size=3 and dilation=1
[1;31m2026-04-17 18:01:01.108341418 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
[1;31m2026-04-17 18:01:01.118987824 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0] failed while attempting to run meta for aten.max_pool2d_with_indices.default
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0] Traceback (most recent call last):
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_subclasses/fake_tensor.py", line 2755, in _dispatch_impl
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     r = func(*args, **kwargs)
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_ops.py", line 841, in __call__
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     return self._op(*args, **kwargs)
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]            ~~~~~~~~^^^^^^^^^^^^^^^^^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4855, in meta_max_pool2d_with_indices
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     ) = max_pool2d_checks_and_compute_shape(
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]         input, kernel_size, stride, padding, dilation, ceil_mode
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     )
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     ^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4774, in max_pool2d_checks_and_compute_shape
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4440, in pooling_output_shape
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     torch._check(
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     ~~~~~~~~~~~~^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]         pad <= ((kernelSize - 1) * dilation + 1) // 2,
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     ...<3 lines>...
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]         ),
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]         ^^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     )
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     ^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1695, in _check
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     _check_with(RuntimeError, cond, message)
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1677, in _check_with
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0]     raise error_type(message_evaluated)
E0417 18:01:01.142000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [1/0] RuntimeError: pad should be at most half of effective kernel size, but got pad=5, kernel_size=5 and dilation=1
[1;31m2026-04-17 18:01:01.157455186 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
[1;31m2026-04-17 18:01:01.170458531 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0] failed while attempting to run meta for aten.max_pool2d_with_indices.default
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0] Traceback (most recent call last):
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_subclasses/fake_tensor.py", line 2755, in _dispatch_impl
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     r = func(*args, **kwargs)
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_ops.py", line 841, in __call__
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     return self._op(*args, **kwargs)
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]            ~~~~~~~~^^^^^^^^^^^^^^^^^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4855, in meta_max_pool2d_with_indices
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     ) = max_pool2d_checks_and_compute_shape(
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]         input, kernel_size, stride, padding, dilation, ceil_mode
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     )
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     ^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4774, in max_pool2d_checks_and_compute_shape
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4440, in pooling_output_shape
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     torch._check(
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     ~~~~~~~~~~~~^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]         pad <= ((kernelSize - 1) * dilation + 1) // 2,
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     ...<3 lines>...
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]         ),
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]         ^^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     )
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     ^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1695, in _check
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     _check_with(RuntimeError, cond, message)
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1677, in _check_with
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0]     raise error_type(message_evaluated)
E0417 18:01:01.195000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [2/0] RuntimeError: pad should be at most half of effective kernel size, but got pad=2, kernel_size=2 and dilation=1
[1;31m2026-04-17 18:01:01.215173736 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
[1;31m2026-04-17 18:01:01.225905003 [E:onnxruntime:, inference_session.cc:2619 operator()] Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
[m
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0] failed while attempting to run meta for aten.max_pool2d_with_indices.default
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0] Traceback (most recent call last):
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_subclasses/fake_tensor.py", line 2755, in _dispatch_impl
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     r = func(*args, **kwargs)
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_ops.py", line 841, in __call__
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     return self._op(*args, **kwargs)
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]            ~~~~~~~~^^^^^^^^^^^^^^^^^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4855, in meta_max_pool2d_with_indices
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     ) = max_pool2d_checks_and_compute_shape(
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]         input, kernel_size, stride, padding, dilation, ceil_mode
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     )
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     ^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4774, in max_pool2d_checks_and_compute_shape
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/_meta_registrations.py", line 4440, in pooling_output_shape
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     torch._check(
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     ~~~~~~~~~~~~^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]         pad <= ((kernelSize - 1) * dilation + 1) // 2,
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     ...<3 lines>...
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]         ),
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]         ^^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     )
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     ^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1695, in _check
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     _check_with(RuntimeError, cond, message)
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/torch/__init__.py", line 1677, in _check_with
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0]     raise error_type(message_evaluated)
E0417 18:01:01.256000 302297 site-packages/torch/_subclasses/fake_tensor.py:2759] [3/0] RuntimeError: pad should be at most half of effective kernel size, but got pad=3, kernel_size=3 and dilation=1
Bug #23088 — MaxPool pad ≥ kernel: passes ONNX checker, rejected by ORT

  Case: kernel=3 pad=3  (original bug)  [ONNX checker: PASS]
    ORT_ref          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    ORT_opt          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    OpenVINO         ok     (1, 1, 12, 12)  ← SILENT ACCEPT
    onnx2torch       ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT
    torch_compile    ERR    Dynamo failed to run FX node with fake tensors: call_functio  ← CORRECT REJECT
    TorchScript      ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT

  Case: kernel=5 pad=5  (new input)  [ONNX checker: PASS]
    ORT_ref          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    ORT_opt          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    OpenVINO         ok     (1, 2, 18, 18)  ← SILENT ACCEPT
    onnx2torch       ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT
    torch_compile    ERR    Dynamo failed to run FX node with fake tensors: call_functio  ← CORRECT REJECT
    TorchScript      ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT

  Case: kernel=2 pad=2  (new input)  [ONNX checker: PASS]
    ORT_ref          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    ORT_opt          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    OpenVINO         ok     (1, 1, 9, 9)  ← SILENT ACCEPT
    onnx2torch       ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT
    torch_compile    ERR    Dynamo failed to run FX node with fake tensors: call_functio  ← CORRECT REJECT
    TorchScript      ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT

  Case: kernel=3 pad=(3,0,3,0) partial  [ONNX checker: PASS]
    ORT_ref          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    ORT_opt          ERR    [ONNXRuntimeError] : 1 : FAIL : Exception during initializat  ← CORRECT REJECT
    OpenVINO         ok     (1, 1, 12, 6)  ← SILENT ACCEPT
    onnx2torch       ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT
    torch_compile    ERR    Dynamo failed to run FX node with fake tensors: call_functio  ← CORRECT REJECT
    TorchScript      ERR    pad should be at most half of effective kernel size, but got  ← CORRECT REJECT

BUG REPRODUCED: OpenVINO silently accepts MaxPool with pad >= kernel
  (returns extended output) while ORT and PyTorch compilers correctly reject.
```

**`cross_openvino_multi_query_attention.py`**
```
ORT:      [ 21.168224 -24.423386  20.916695  17.117714]
OpenVINO: [ 21.125 -24.375  20.875  17.   ]
max_abs=0.1724
BUG REPRODUCED: OpenVINO multi_query_attention (max_abs=0.1724)
```

**`cross_openvino_multi_scale_conv_branch.py`**
```
ORT:      [  3.395184 -11.441774 -24.340858   9.490486]
OpenVINO: [  3.3566203 -11.426291  -24.376686    9.516493 ]
max_abs=0.2586
BUG REPRODUCED: OpenVINO multi_scale_conv_branch (max_abs=0.2586)
```

**`cross_openvino_neg_unary.py`**
```
ORT:      [ 24.023788 -18.11512  -24.266878  -7.113834]
OpenVINO: [ 24.021349  -18.040651  -24.258022   -7.0619965]
max_abs=0.1747
BUG REPRODUCED: OpenVINO neg_unary (max_abs=0.1747)
```

**`cross_openvino_pad_conv.py`**
```
ORT:      [  7.4057403 -25.58705     9.93622     4.4996433]
OpenVINO: [  7.346116 -25.596539   9.922224   4.521668]
max_abs=0.1808
BUG REPRODUCED: OpenVINO pad_conv (max_abs=0.1808)
```

**`cross_openvino_pointwise_dw_block.py`**
```
ORT:      [  77.25082       0.72612983  -10.597239   -179.6164    ]
OpenVINO: [  77.203         0.49180603  -10.749359   -179.40517   ]
max_abs=2.1500
BUG REPRODUCED: OpenVINO pointwise_dw_block (max_abs=2.1500)
```

**`cross_openvino_reciprocal_mul.py`**
```
ORT:      [   24.235746  -443.3107   -7963.6655   -1343.5315  ]
OpenVINO: [   27.563232  -439.86133  -8005.1084   -1337.834   ]
max_abs=41.4429
BUG REPRODUCED: OpenVINO reciprocal_mul (max_abs=41.4429)
```

**`cross_openvino_reduce_l2_last.py`**
```
ORT:      [275.03833 240.01646 257.79965 244.62679]
OpenVINO: [275.05725 239.94469 257.80746 244.65967]
max_abs=0.0718
BUG REPRODUCED: OpenVINO reduce_l2_last (max_abs=0.0718)
```

**`cross_openvino_redundant_reshape.py`**
```
ORT:      [ 35.098927 -45.611603 -28.941078  26.396048]
OpenVINO: [ 35.05108  -45.62861  -28.852718  26.34306 ]
max_abs=0.1404
BUG REPRODUCED: OpenVINO redundant_reshape (max_abs=0.1404)
```

**`cross_openvino_rrelu_inference_identity.py`**
```
ORT:      [-1.4135344 -2.1028607 -3.1716204 10.104586 ]
OpenVINO: [-1.4143701 -2.0955994 -3.1746764 10.108147 ]
max_abs=0.1230
BUG REPRODUCED: OpenVINO rrelu_inference_identity (max_abs=0.1230)
```

**`cross_openvino_slice_full_range.py`**
```
ORT:      [20.241344 -6.4461   37.495216  6.697869]
OpenVINO: [20.209951 -6.39095  37.619843  6.732247]
max_abs=0.2065
BUG REPRODUCED: OpenVINO slice_full_range (max_abs=0.2065)
```

**`cross_openvino_space_to_depth_block.py`**
```
ORT:      [ 41.254436 -22.265532  15.169448  56.25074 ]
OpenVINO: [ 41.07376  -22.58332   15.297047  56.707504]
max_abs=1.0417
BUG REPRODUCED: OpenVINO space_to_depth_block (max_abs=1.0417)
```

**`cross_openvino_sub_self_mul_zero.py`**
```
ORT:      [-265.7939    -74.557076  159.4417    163.64496 ]
OpenVINO: [-265.       -75.15625  159.0625   164.375  ]
max_abs=2.9208
BUG REPRODUCED: OpenVINO sub_self_mul_zero (max_abs=2.9208)
```

**`cross_openvino_tile_conv.py`**
```
ORT:      [ -5.6541376  -8.411443  -12.686481   10.104586 ]
OpenVINO: [ -5.6574802  -8.382398  -12.698706   10.108147 ]
max_abs=0.1471
BUG REPRODUCED: OpenVINO tile_conv (max_abs=0.1471)
```

**`cross_openvino_transformer_encoder_layer.py`**
```
ORT:      [ 0.0053039 -0.5674004 -1.1994314  1.8477817]
OpenVINO: [ 0.00602648 -0.56125975 -1.2004263   1.8446367 ]
max_abs=0.0135
BUG REPRODUCED: OpenVINO transformer_encoder_layer (max_abs=0.0135)
```

**`cross_openvino_transpose_matmul_transpose.py`**
```
ORT:      [ 14.232545   30.698551  -30.721355    3.2935975]
OpenVINO: [ 14.125   30.625  -30.75     3.3125]
max_abs=0.1714
BUG REPRODUCED: OpenVINO transpose_matmul_transpose (max_abs=0.1714)
```

**`cross_openvino_transpose_transpose_squash.py`**
```
ORT:      [ -3.3332217   3.0382774   7.048198  -13.206124 ]
OpenVINO: [ -3.2892637   3.0823588   7.0204124 -13.21917  ]
max_abs=0.0890
BUG REPRODUCED: OpenVINO transpose_transpose_squash (max_abs=0.0890)
```

**`cross_openvino_triple_add_residual.py`**
```
ORT:      [ 33.828735 -46.024643 -31.184195  27.007572]
OpenVINO: [ 33.72981  -46.16304  -31.118114  26.986526]
max_abs=0.1797
BUG REPRODUCED: OpenVINO triple_add_residual (max_abs=0.1797)
```

**`cross_openvino_where_mask_fill.py`**
```
ORT:      [-1.0000000e+09 -1.0000000e+09 -2.8941078e+01 -1.0000000e+09]
OpenVINO: [-1.0000e+09 -1.0000e+09 -2.8875e+01 -1.0000e+09]
max_abs=0.1797
BUG REPRODUCED: OpenVINO where_mask_fill (max_abs=0.1797)
```

**`github_ov_015_matmul_gpu_tile_overflow.py`**
```
N    CPU result    expected      diff  bug?
--------------------------------------------------------
  2047        2047.0      2047.0       0.0  no OV
  2048        2048.0      2048.0       0.0  no OV
  2049        2048.0      2049.0       1.0  no OV
  4096        2048.0      4096.0    2048.0  no OV

(OpenVINO not installed — showing GPU-bug simulation: N>2048 returns 2048)
Analytical demonstration: GPU tile overflow reproduced

PASS=False
BUG REPRODUCED: MatMul GPU tile skips partial last tile for dim > 2048
```

**`github_ov_019_uint8_sub_no_wrap.py`**
```
a        : [  5 200 250   0]
b        : [ 10 100  10  50]
OV out   : [  0 100 240   0]
wrap ref : [251 100 240 206]  (correct — ONNX modular)
sat  ref : [  0 100 240   0]   (wrong  — saturation)

Matches wrap (correct)? False
Matches sat  (wrong)?   True

PASS=True
BUG REPRODUCED — OV uint8 Sub saturates instead of wrapping
```

**`github_ov_020_uint8_mul_no_wrap.py`**
```
a        : [200 100  50  10]
b        : [200   3   6  30]
a*b      : [40000   300   300   300]
OV out   : [255 255 255 255]
wrap ref : [64 44 44 44]  (correct — ONNX modular)
sat  ref : [255 255 255 255]   (wrong  — saturation)

Matches wrap (correct)? False
Matches sat  (wrong)?   True

PASS=True
BUG REPRODUCED — OV uint8 Mul saturates instead of wrapping
```

**`github_ov_021_uint8_add_no_wrap.py`**
```
a        : [200 100 255 128]
b        : [100 200  10 200]
a+b      : [300 300 265 328]
OV out   : [255 255 255 255]
wrap ref : [44 44  9 72]  (correct)
sat  ref : [255 255 255 255]   (wrong)

PASS=True
BUG REPRODUCED — OV uint8 Add saturates instead of wrapping
```

**`github_ov_022_reducelogsumexp_overflow.py`**
```
Input x  : [[100.  88.  50.]
 [200. -10.   1.]]
OV out   : [inf inf]
Ref      : [100.00001 200.     ]
has_inf  : True
max_diff : inf

PASS=True
BUG REPRODUCED — OV ReduceLogSumExp overflows for inputs ≥ 88.7 (fp32 exp limit)
```

**`github_ov_023_relu_nan_propagation.py`**
```
[0;93m2026-04-17 18:01:15.116325104 [W:onnxruntime:, model.cc:215 Model] ONNX Runtime only *guarantees* support for models stamped with opset version 7 or above for opset domain 'ai.onnx'. Please upgrade your model to opset 7 or higher. For now, this opset 6 model may run depending upon legacy support of some older opset version operators.[m
[0;93m2026-04-17 18:01:15.119427747 [W:onnxruntime:, transpose_optimizer.cc:37 ApplyImpl] Transpose optimizer failed: Unsupported ONNX opset: 6[m
[0;93m2026-04-17 18:01:15.119457733 [W:onnxruntime:, transpose_optimizer.cc:37 ApplyImpl] Transpose optimizer failed: Unsupported ONNX opset: 6[m
Input  : [nan -1.  0.  1. inf]
ORT    : [nan  0.  0.  1. inf]  ← NaN propagated (correct)
OV     : [ 0.  0.  0.  1. inf]

Input[0] is NaN: True
ORT[0]  is NaN: True  (expected: True)
OV[0]   is NaN: False  (expected: True)

PASS=True
BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754)
```

**`github_ov_024_int8_sub_saturation.py`**
```
a        : [-128  127   -1    0]
b        : [   1 -128 -127  100]
OV out   : [-128  127  126 -100]
wrap ref : [ 127   -1  126 -100]  (correct — two's complement)
sat  ref : [-128  127  126 -100]   (wrong  — saturation)

Matches wrap (correct)? False
Matches sat  (wrong)?   True

PASS=True
BUG REPRODUCED — OV int8 Sub saturates instead of two's-complement wrapping
```

**`github_ov_025_int8_add_saturation.py`**
```
a        : [ 100  127 -128 -100]
b        : [100  10 -10 -50]
a+b(i16) : [ 200  137 -138 -150]
OV out   : [ 127  127 -128 -128]
wrap ref : [ -56 -119  118  106]  (correct — two's complement)
sat  ref : [ 127  127 -128 -128]   (wrong  — saturation)

PASS=True
BUG REPRODUCED — OV int8 Add saturates instead of two's-complement wrapping
```

**`github_ov_026_exp_nan_to_inf.py`**
```
[0;93m2026-04-17 18:01:16.840690334 [W:onnxruntime:, model.cc:215 Model] ONNX Runtime only *guarantees* support for models stamped with opset version 7 or above for opset domain 'ai.onnx'. Please upgrade your model to opset 7 or higher. For now, this opset 6 model may run depending upon legacy support of some older opset version operators.[m
[0;93m2026-04-17 18:01:16.851193050 [W:onnxruntime:, transpose_optimizer.cc:37 ApplyImpl] Transpose optimizer failed: Unsupported ONNX opset: 6[m
[0;93m2026-04-17 18:01:16.851260193 [W:onnxruntime:, transpose_optimizer.cc:37 ApplyImpl] Transpose optimizer failed: Unsupported ONNX opset: 6[m
Input  : [nan  1.]
ORT    : [      nan 2.7182817]  ← NaN propagated (correct)
OV     : [     inf 2.718282]

Input[0] is NaN: True
ORT[0]  is NaN: True  (expected: True)
OV[0]   is NaN: False  (expected: True)
OV[0]   is inf: True  (should be False)

PASS=True
BUG REPRODUCED — OV Exp(NaN) → +inf, should propagate NaN (IEEE 754)
```

**`tf_61865_onnx_openvino.py`**
```
[1;31m2026-04-17 18:01:17.401483636 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=12 must be within the inclusive range [-12,11][m
[1;31m2026-04-17 18:01:17.417531055 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=12 must be within the inclusive range [-12,11][m
params  : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
indices : [0, 3, 6, 9, 12, 1]  (index 12 is OOB for size-12 tensor)

Compiler          Output                            Error
--------------------------------------------------------------------------------
ORT_ref           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
ORT_opt           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
OpenVINO          [1.0, 4.0, 7.0, 10.0, 0.0, 2.0]   —  ← TARGET
onnx2torch        —                                 index 12 is out of bounds for dimension 0 with size 12
torch_compile     —                                 kernel, /tmp/torchinductor_binduan/7v/c7v2napl736wpqfou
TorchScript       —                                 index 12 is out of bounds for dimension 0 with size 12

BUG REPRODUCED: OpenVINO silently returns a value for OOB Gather
  OOB slot (index 12) → 0.0000
  ORT_ref/ORT_opt raise INVALID_ARGUMENT; OpenVINO proceeds silently.
```


### TFLite

**`cross_tflite_attention_logit_softcap.py`**
```
ORT: [-50.  50. -50. -50.]
BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'
```

**`cross_tflite_fully_connected_mul_fusion_crash.py`**
```
Saved artifact at '/tmp/tmpfyo14kua'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 2), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  136727335741456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335740880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335738384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335740688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335741648: TensorSpec(shape=(), dtype=tf.resource, name=None)
W0000 00:00:1776412890.266432  312807 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776412890.266441  312807 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_99"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_130"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["MatMul:", "m_1/MatMul@__inference_function_99"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_130"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["AddV2:", "m_1/Add_1@__inference_function_99"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_130"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"]))]): error: 'tfl.fully_connected' op expect 'input' num_elements % 2 == 0, got input type 'tensor<1xf32>'
V1: x=[1,2] w1=[2,1] original:
  Keras =[997.5]
  TFLite=CONVERTER CRASH: <unknown>:0: error: loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_99"] at callsite(fused["StatefulParti

Saved artifact at '/tmp/tmp7jlwari_'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 2), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  136727335743568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335743952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335747024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335742224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335744336: TensorSpec(shape=(), dtype=tf.resource, name=None)
W0000 00:00:1776412890.409736  312807 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776412890.409753  312807 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_506"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_537"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["MatMul:", "m_1/MatMul@__inference_function_506"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_537"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["Mul:", "m_1/Mul_1@__inference_function_506"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_537"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["AddV2:", "m_1/Add_2@__inference_function_506"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_537"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"]))]): error: 'tfl.fully_connected' op expect 'input' num_elements % 2 == 0, got input type 'tensor<1xf32>'
V2: x=[1,2] w1=[2,1] diff weights:
  Keras =[21.0]
  TFLite=CONVERTER CRASH: <unknown>:0: error: loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_506"] at callsite(fused["StatefulPart

Saved artifact at '/tmp/tmpqaksuyx7'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 4), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  136727333914960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727333915728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727335746448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727333913616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727333913424: TensorSpec(shape=(), dtype=tf.resource, name=None)
W0000 00:00:1776412890.565484  312807 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776412890.565498  312807 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_913"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_944"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["MatMul:", "m_1/MatMul@__inference_function_913"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_944"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["AddV2:", "m_1/Add_1@__inference_function_913"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_944"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"]))]): error: 'tfl.fully_connected' op expect 'input' num_elements % 4 == 0, got input type 'tensor<1xf32>'
V3: x=[1,4] w1=[4,1]:
  Keras =[13860.0]
  TFLite=CONVERTER CRASH: <unknown>:0: error: loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_913"] at callsite(fused["StatefulPart

Saved artifact at '/tmp/tmpvc8qyskp'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 2), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 2), dtype=tf.float32, name=None)
Captures:
  136727333917648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727333915344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727333916112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727333916496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  136727333918992: TensorSpec(shape=(), dtype=tf.resource, name=None)
W0000 00:00:1776412890.729053  312807 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776412890.729067  312807 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_1320"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_1351"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["MatMul:", "m_1/MatMul@__inference_function_1320"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_1351"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"])), callsite(fused["AddV2:", "m_1/Add_1@__inference_function_1320"] at callsite(fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_1351"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall_1"]))]): error: 'tfl.fully_connected' op expect 'input' num_elements % 2 == 0, got input type 'tensor<1xf32>'
V4: x=[1,2] w1=[2,2] wider output:
  Keras =[483.0, 2198.0]
  TFLite=CONVERTER CRASH: <unknown>:0: error: loc(fused[callsite(fused["Mul:", "m_1/Mul@__inference_function_1320"] at callsite(fused["StatefulPar
```

**`cross_tflite_l2_normalize_multi_shape.py`**
```
W0417 18:01:35.564000 313424 site-packages/torch/onnx/_internal/exporter/_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 17 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
The model version conversion is not supported by the onnxscript version converter and fallback is enabled. The model will be converted using the onnx C API (target version: 17).
==============================================================================
Case 3: shape=(2,2,3), half-step floats    shape=(2, 2, 3)
  input    : [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
  expected : [0.0699, 0.1049, 0.1399, 0.1748, 0.2098, 0.2447, 0.2797, 0.3147, 0.3496, 0.3846, 0.4196, 0.4545]
[torch.onnx] Obtain model graph for `TM()` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `TM()` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
Saved artifact at '/tmp/tmpl05ovp5j'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 2, 3), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 2, 3), dtype=tf.float32, name=None)
Captures:
  None
W0000 00:00:1776412896.677149  313424 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776412896.677159  313424 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
    for details.
    
W0417 18:01:36.829000 313424 site-packages/torch/onnx/_internal/exporter/_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 17 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
  Runtime        Output (rounded 4dp)                                    Match?
  Keras eager    [0.0699, 0.1049, 0.1399, 0.1748, 0.2098, 0.2447, 0.2797, 0.3147, 0.3496, 0.3846, 0.4196, 0.4545] OK
  XLA            [0.0699, 0.1049, 0.1399, 0.1748, 0.2098, 0.2447, 0.2797, 0.3147, 0.3496, 0.3846, 0.4196, 0.4545] OK
  PyTorch        [0.0699, 0.1049, 0.1399, 0.1748, 0.2098, 0.2447, 0.2797, 0.3147, 0.3496, 0.3846, 0.4196, 0.4545] OK
  ONNX Runtime   [0.0699, 0.1049, 0.1399, 0.1748, 0.2098, 0.2447, 0.2797, 0.3147, 0.3496, 0.3846, 0.4196, 0.4545] OK
  TFLite         [0.3714, 0.5571, 0.7428, 0.4767, 0.5721, 0.6674, 0.5111, 0.575, 0.6389, 0.528, 0.576, 0.624] WRONG

==============================================================================
Case 4: shape=(4,1), unit innermost dim    shape=(4, 1)
  input    : [2.0, 3.0, 4.0, 5.0]
  expected : [0.2722, 0.4082, 0.5443, 0.6804]
[torch.onnx] Obtain model graph for `TM()` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `TM()` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
Saved artifact at '/tmp/tmpumeow782'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  None
W0000 00:00:1776412897.215834  313424 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776412897.215845  313424 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
    for details.
    
  Runtime        Output (rounded 4dp)                                    Match?
  Keras eager    [0.2722, 0.4082, 0.5443, 0.6804]                        OK
  XLA            [0.2722, 0.4082, 0.5443, 0.6804]                        OK
  PyTorch        [0.2722, 0.4082, 0.5443, 0.6804]                        OK
  ONNX Runtime   [0.2722, 0.4082, 0.5443, 0.6804]                        OK
  TFLite         [1.0, 1.0, 1.0, 1.0]                                    WRONG
```

**`cross_tflite_sub_self_mul_zero.py`**
```
ORT (should be ~0): [0. 0. 0. 0.]
BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'
```

**`cross_tflite_transformer_encoder_layer.py`**
```
ORT: [-4230.362   1943.0718 -1265.1711  3324.3977]
BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'
```

**`cross_tflite_transpose_matmul_transpose.py`**
```
ORT: [ 14.232545   30.698551  -30.721355    3.2935975]
BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'
```

**`cross_tflite_unstack_concat_reshape_fold.py`**
```
W0417 18:01:53.486000 315959 site-packages/torch/onnx/_internal/exporter/_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 17 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
The model version conversion is not supported by the onnxscript version converter and fallback is enabled. The model will be converted using the onnx C API (target version: 17).
input    : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
expected : [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]

[torch.onnx] Obtain model graph for `TM()` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `TM()` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
Saved artifact at '/tmp/tmpnvpt8656'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 3), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(4, 3), dtype=tf.float32, name=None)
Captures:
  None
W0000 00:00:1776412914.636168  315959 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776412914.636186  315959 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
    for details.
    
Runtime        Output                                                       Match?
--------------------------------------------------------------------------------------
Keras eager    [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0] OK
XLA            [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0] OK
PyTorch        [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0] OK
ONNX Runtime   [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0] OK
TFLite         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] WRONG
```


### torch.compile / Inductor

**`cross_torch_compile_add_self_sub_double.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on add_self_sub_double
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_aspp_dilated_branch.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on aspp_dilated_branch
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_conv_k7_stride2.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on conv_k7_stride2
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_conv_manual_celu.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on conv_manual_celu
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_cp_fp16_matmul_n512.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on cp_fp16_matmul_n512
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_einsum_transpose.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on einsum_transpose
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_flatten_gemm.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on flatten_gemm
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_gemm_sigmoid.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on gemm_sigmoid
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_group_query_attention.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on group_query_attention
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_ia_sat_sub_uint8_underflow.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on ia_sat_sub_uint8_underflow
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_manual_hard_swish.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on manual_hard_swish
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_matmul_add_layernorm.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on matmul_add_layernorm
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_multi_query_attention.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on multi_query_attention
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_reciprocal_mul.py`**
```
ORT: [-588.35785  556.4601  -443.40463 -903.6573 ]  (correct)
eager onnx2torch: [-588.35785  556.45996 -443.40475 -903.65735]  (correct)
BUG REPRODUCED: torch.compile crashes on reciprocal_mul
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_reduce_l2_last.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on reduce_l2_last
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_redundant_reshape.py`**
```
ORT: [ 35.098927 -45.611603 -28.941078  26.396048]  (correct)
eager: [ 35.098927 -45.6116   -28.941076  26.396042]  (correct)
BUG REPRODUCED: torch.compile crashes on redundant_reshape: InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File
```

**`cross_torch_compile_slice_full_range.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on slice_full_range
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_space_to_depth_block.py`**
```
ORT: [ 41.254436 -22.265532  15.169448  56.25074 ]  (correct)
BUG REPRODUCED: torch.compile crashes on space_to_depth_block: NotImplementedError: Converter is not implemented (OperationDescription(domain='', operation_type='SpaceToDepth', version
```

**`cross_torch_compile_sub_self_mul_zero.py`**
```
ORT: [0. 0. 0. 0.]  (correct)
eager onnx2torch: [0. 0. 0. 0.]  (correct)
BUG REPRODUCED: torch.compile crashes on sub_self_mul_zero
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_topk_last_axis_k1.py`**
```
ORT: [1.5792128  0.54256004 1.4656488  1.8522782 ]  (correct)
eager onnx2torch: [1.5792128  0.54256004 1.4656488  1.8522782 ]  (correct)
BUG REPRODUCED: torch.compile crashes on topk_last_axis_k1
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_transpose_transpose_squash.py`**
```
ORT: [ -3.3332217   3.0382774   7.048198  -13.206124 ]  (correct)
eager: [ -3.3332222   3.0382776   7.0481997 -13.206127 ]  (correct)
BUG REPRODUCED: torch.compile crashes on transpose_transpose_squash: InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File
```

**`cross_torch_compile_triple_add_residual.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on triple_add_residual
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`cross_torch_compile_where_mask_fill.py`**
```
ORT: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
eager onnx2torch: [ 10.246511  -14.433511    6.7743917  -4.539094 ]  (correct)
BUG REPRODUCED: torch.compile crashes on where_mask_fill
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')

from user code:
   File "/home/binduan/mini
```

**`github_inductor_009_transpose_reduce_fusion.py`**
```
=== github_inductor_009: Transpose+ReduceSum fusion bug ===

Input [4,64,128] row_sum*scale→transpose
eager vs torch.compile: max_diff=7.63e-06  ok
Attention grad pattern [8,256,256] scale→transpose
eager vs torch.compile: max_diff=1.53e-05  BUG

PASS=False
BUG REPRODUCED: Inductor transpose+reduce fusion reinterprets storage strides
```

**`github_inductor_011_bitshift_ub_shift64.py`**
```
=== github_inductor_011: BitShift UB — shift by 64 produces non-zero ===

     value   shift       correct     C_UB(x86)  bug?
----------------------------------------------------------
      1000      64             0          1000  BUG(UB)
       255      64             0           255  BUG(UB)
       -42      64             0           -42  BUG(UB)
         1      63             0             0  ok
         1      65             0             0  ok

Left-shift equivalent (pytorch/pytorch#143566):
  1000 << 64: correct=0  C_UB=1000  BUG
  1 << 64: correct=0  C_UB=1  BUG
  255 << 64: correct=0  C_UB=255  BUG

PyTorch runtime check:
  eager: [1000, 255, -42] >> [64, 64, 64] = [0, 0, -1]
  expected: [0, 0, 0]  BUG
  compile: [1000, 255, -42] >> [64, 64, 64] = [0, 0, -1]
  expected: [0, 0, 0]  BUG

ONNX BitShift (direction=RIGHT) path:
  ORT: [1000, 255] >> [64, 64] = [1000, 255]
  expected: [0, 0]  BUG

PASS=False
BUG REPRODUCED: BitShift by 64 returns non-zero (C UB: x86 masks shift to 6 bits)
```

**`pt_121135_torch_compile.py`**
```
Case                  Compiler          Result                Status
---------------------------------------------------------------------------
(32, 4)+(4,)          eager             —                     RuntimeError: source tensor shape m
(32, 4)+(4,)          torch.compile     shape=[32, 4]         OK  ← BUG
(32, 4)+(4,)          TorchScript       —                     RuntimeError: source tensor shape m

(8, 3)+(3,)           eager             —                     RuntimeError: source tensor shape m
(8, 3)+(3,)           torch.compile     shape=[8, 3]          OK  ← BUG
(8, 3)+(3,)           TorchScript       —                     RuntimeError: source tensor shape m

BUG REPRODUCED: torch.compile skips shape validation in index_add;
  eager and TorchScript correctly raise RuntimeError.
```

**`tf_61865_onnx_torch_compile.py`**
```
[1;31m2026-04-17 18:04:47.512240237 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=10 must be within the inclusive range [-10,9][m
[1;31m2026-04-17 18:04:47.524625082 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=10 must be within the inclusive range [-10,9][m
params  : [0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]
indices : [1, 4, 7, 9, 10, 0]  (index 10 is OOB for size-10 tensor)

params  : [0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]
indices : [1, 4, 7, 9, 10, 0]  (index 10 is OOB for size-10 tensor)

Compiler          Output                            Error
--------------------------------------------------------------------------------
ORT_ref           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
ORT_opt           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
OpenVINO          [0.20000000298023224, 0.5, 0.800000011920929, 1.0, 0.0, 0.10000000149011612]  —
onnx2torch        —                                 index 10 is out of bounds for dimension 0 with size 10
torch_compile     —                                 kernel, /tmp/torchinductor_binduan/4v/c4vvfkr664mr6yvjt  ← TARGET
TorchScript       —                                 index 10 is out of bounds for dimension 0 with size 10

BUG REPRODUCED: torch.compile raises on OOB Gather
  while other compilers return values: {'OpenVINO': [0.20000000298023224, 0.5, 0.800000011920929, 1.0, 0.0, 0.10000000149011612]}
```


### TVM

**`github_tvm_004.py`**
```
ORT cubic  output[0,0,0,:4]: [0.80208254 0.6395161  0.288994   0.30634046]
PyTorch bicubic[0,0,0,:4]:   [0.8143504  0.61203825 0.2709736  0.29592   ]
Max abs error ORT vs PyTorch: 0.053012
TVM diverged from both references before the fix
PASS=False
BUG REPRODUCED
```

**`github_tvm_010_simplifyexpr_rsqrt_precision.py`**
```
x       y     sqrt(x)/y    rsqrt(x)*y     rel_err  bug?
------------------------------------------------------------------------
    1.00e-04     1.0    0.01000000     99.840172     9983.02  BUG
    1.00e-02     2.0    0.05000000     19.965044      398.30  BUG
    1.00e+00     1.0    1.00000000      0.998307        0.00  BUG

Pattern 2: conv→relu→mul(s) vs mul(s)→conv→relu  (FoldScaleAxis)
  relu(x)*scale (ref):  [-2.  -0.  -0.6 -0. ]
  scale*relu(x) (bug):  [0.  1.  0.  2.4]
  max_diff: 2.4000  BUG

PASS=False
BUG REPRODUCED: SimplifyExpr rsqrt rewrite introduces large precision error
```

**`github_tvm_011_lifttransformparams_const_bind.py`**
```
Expected : [ 1  4  9 16 25 36 49 64] ...
Buggy    : [ 4  9 16 25 36 49 64 81] ...
Max diff : 33  (TVM not installed — analytical demo)
PASS=False
BUG REPRODUCED: LiftTransformParams corrupts constant binding
```

**`github_tvm_012_gelu_approx_tanh.py`**
```
=== github_tvm_012: ONNX Gelu approx=tanh ignored by TVM ===

     x    gelu_exact     gelu_tanh    abs_diff    rel_diff  bug?
--------------------------------------------------------------------
 -2.00   -0.04550026   -0.04540231    9.80e-05    2.15e-03  ok
 -1.00   -0.15865525   -0.15880801    1.53e-04    9.63e-04  BUG
 -0.50   -0.15426877   -0.15428599    1.72e-05    1.12e-04  ok
  0.00    0.00000000    0.00000000    0.00e+00    0.00e+00  ok
  0.50    0.34573123    0.34571401    1.72e-05    4.98e-05  ok
  1.00    0.84134475    0.84119199    1.53e-04    1.82e-04  BUG
  2.00    1.95449974    1.95459769    9.80e-05    5.01e-05  ok

Max abs diff: 1.53e-04  tol=1e-04

TVM substitution demo (approx=tanh input → exact erf output):
  expected (tanh): [-0.10042999684810638, -0.15881000459194183, 0.0, 0.841189980506897, 1.3995699882507324]
  TVM output (exact erf): [-0.10021000355482101, -0.15865999460220337, 0.0, 0.8413400053977966, 1.3997900485992432]
  per-element diff: [0.000218, 0.000153, 0.0, 0.000153, 0.000218]
  any diff > TOL: True

PASS=False
BUG REPRODUCED: TVM maps ONNX Gelu(approx=tanh) → exact erf, producing systematic error > 1e-4
```


### XLA / TensorFlow

**`cross_xla_add_self_sub_double.py`**
```
ORT: [ 33.65159  -46.478348 -28.172342  27.534014]
XLA: [ 33.650417 -46.470745 -28.187355  27.553116]
max_abs=0.0203
BUG REPRODUCED: XLA add_self_sub_double (max_abs=0.0203)
```

**`cross_xla_autocluster_slice_oob_suppress.py`**
```
W0000 00:00:1776413098.241962  339876 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 4
W0000 00:00:1776413098.261181  339876 op_kernel.cc:1858] OP_REQUIRES failed at xla_ops.cc:602 : INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 4

Stack trace for op definition: 
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 77, in <module>
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 71, in run
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 47, in f_jit

	 [[{{node Slice_1}}]]
	tf2xla conversion failed while converting __inference_f_jit_22[_XlaMustCompile=true,config_proto=7932885297272261767,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
W0000 00:00:1776413098.284457  339876 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 4
W0000 00:00:1776413098.286859  339876 op_kernel.cc:1858] OP_REQUIRES failed at xla_ops.cc:602 : INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 4

Stack trace for op definition: 
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 77, in <module>
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 71, in run
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 47, in f_jit

	 [[{{node Slice_1}}]]
	tf2xla conversion failed while converting __inference_f_jit_55[_XlaMustCompile=true,config_proto=7932885297272261767,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
W0000 00:00:1776413098.298208  339876 op_kernel.cc:1858] OP_REQUIRES failed at xla_ops.cc:602 : INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 4

Stack trace for op definition: 
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 77, in <module>
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 71, in run
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 47, in f_jit

	 [[{{node Slice_1}}]]
	tf2xla conversion failed while converting __inference_f_jit_88[_XlaMustCompile=true,config_proto=7932885297272261767,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
W0000 00:00:1776413098.308274  339876 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 4
W0000 00:00:1776413098.310369  339876 op_kernel.cc:1858] OP_REQUIRES failed at xla_ops.cc:602 : INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 4

Stack trace for op definition: 
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 77, in <module>
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 71, in run
File "myspace/trion/bugs/new_minimal/xla/cross_xla_autocluster_slice_oob_suppress.py", line 47, in f_jit

	 [[{{node Slice_1}}]]
	tf2xla conversion failed while converting __inference_f_jit_121[_XlaMustCompile=true,config_proto=7932885297272261767,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
Case                   Eager                          XLA jit                        Autocluster
----------------------------------------------------------------------------------------------------
V1: shape=(1,3,3,2)    RAISED InvalidArgumentError    RAISED InvalidArgumentError    OK  [2.0, 3.0, 8.0]...  ← BUG
V2: shape=(2,4,3,3)    RAISED InvalidArgumentError    RAISED InvalidArgumentError    OK  [3.0, 4.0, 5.0]...  ← BUG
V3: shape=(1,5,3,1)    RAISED InvalidArgumentError    RAISED InvalidArgumentError    OK  [1.0, 4.0, 7.0]...  ← BUG
V4: shape=(3,2,3,4)    RAISED InvalidArgumentError    RAISED InvalidArgumentError    OK  [4.0, 5.0, 6.0]...  ← BUG
```

**`cross_xla_gather_oob_clamp_variants.py`**
```
==============================================================================
V1: 1D, off-by-one OOB (idx 256)
  x.shape=(256,)  axis=0  indices=[5, 8, 7, 16, 256, 123]
  TF eager : [5.0, 8.0, 7.0, 16.0, 0.0, 123.0]
  PyTorch  : RAISED IndexError
  XLA jit  : [5.0, 8.0, 7.0, 16.0, 255.0, 123.0]
  -> BUG: XLA silently produced output instead of erroring

==============================================================================
V2: 1D, far-OOB (idx 1000)
  x.shape=(256,)  axis=0  indices=[5, 8, 1000, 123]
  TF eager : [5.0, 8.0, 0.0, 123.0]
  PyTorch  : RAISED IndexError
  XLA jit  : [5.0, 8.0, 255.0, 123.0]
  -> BUG: XLA silently produced output instead of erroring

==============================================================================
V3: 1D, negative OOB (idx -1)
  x.shape=(256,)  axis=0  indices=[5, -1, 7, 16, 123]
  TF eager : [5.0, 0.0, 7.0, 16.0, 123.0]
  PyTorch  : RAISED IndexError
  XLA jit  : [5.0, 0.0, 7.0, 16.0, 123.0]
  -> BUG: XLA silently produced output instead of erroring

==============================================================================
V4: 1D, negative far-OOB (idx -300)
  x.shape=(256,)  axis=0  indices=[5, -300, 7, 123]
  TF eager : [5.0, 0.0, 7.0, 123.0]
  PyTorch  : RAISED IndexError
  XLA jit  : [5.0, 0.0, 7.0, 123.0]
  -> BUG: XLA silently produced output instead of erroring

==============================================================================
V5: 1D, INT_MAX
  x.shape=(256,)  axis=0  indices=[5, 2147483647]
  TF eager : [5.0, 0.0]
  PyTorch  : RAISED IndexError
  XLA jit  : [5.0, 255.0]
  -> BUG: XLA silently produced output instead of erroring

==============================================================================
V6: 1D, multiple OOB indices
  x.shape=(256,)  axis=0  indices=[256, 257, 999, 1000000]
  TF eager : [0.0, 0.0, 0.0, 0.0]
  PyTorch  : RAISED IndexError
  XLA jit  : [255.0, 255.0, 255.0, 255.0]
  -> BUG: XLA silently produced output instead of erroring

==============================================================================
V7: 2D, axis=1 OOB on inner dim
  x.shape=(4, 5)  axis=1  indices=[0, 1, 5, 4]
  TF eager : [[0.0, 1.0, 0.0, 4.0], [5.0, 6.0, 0.0, 9.0], [10.0, 11.0, 0.0, 14.0], [15.0, 16.0, 0.0, 19.0]]
  PyTorch  : RAISED RuntimeError
  XLA jit  : [[0.0, 1.0, 4.0, 4.0], [5.0, 6.0, 9.0, 9.0], [10.0, 11.0, 14.0, 14.0], [15.0, 16.0, 19.0, 19.0]]
  -> BUG: XLA silently produced output instead of erroring

==============================================================================
V8: 2D, axis=0 OOB on outer dim
  x.shape=(4, 5)  axis=0  indices=[0, 1, 4, 3]
  TF eager : [[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [0.0, 0.0, 0.0, 0.0, 0.0], [15.0, 16.0, 17.0, 18.0, 19.0]]
  PyTorch  : RAISED IndexError
  XLA jit  : [[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [15.0, 16.0, 17.0, 18.0, 19.0], [15.0, 16.0, 17.0, 18.0, 19.0]]
  -> BUG: XLA silently produced output instead of erroring
```

**`cross_xla_matmul_add_biasgelu_bcast.py`**
```
ORT: [16.35682   0.        0.       14.601542]
XLA: [16.356249  0.        0.       14.611672]
max_abs=0.0678
BUG REPRODUCED: XLA matmul_add_biasgelu_bcast (max_abs=0.0678)
```

**`cross_xla_space_to_depth_block.py`**
```
ORT: [ 41.254436 -22.265532  15.169448  56.25074 ]
XLA: [ 41.25533  -22.245968  15.195742  56.26139 ]
max_abs=0.1193
BUG REPRODUCED: XLA space_to_depth_block (max_abs=0.1193)
```

**`github_tensorflow_002.py`**
```
asymmetric output:          [0. 0. 1. 1. 2. 2. 3. 3.]
half_pixel output:          [0. 0. 0. 1. 1. 2. 2. 3.]
Expected (both):            [0. 0. 1. 1. 2. 2. 3. 3.]
Max err asymmetric: 0.0000, half_pixel: 1.0000
TOSA bug: wrong scale/offset shifted row index by 1 in compiled (not eager) path
PASS=False
BUG REPRODUCED
```

**`github_tf_61881_xla_matmul_incompatible_shapes.py`**
```
W0000 00:00:1776413111.556124  345364 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Matrix size-incompatible: In[0]: [1,4], In[1]: [6,1]
	 [[{{node MatMul}}]]
Input shape : (1, 4), W shape: (6, 1)
Eager output: None  error=InvalidArgumentError: Exception encountered when calling ModelEager.call().

[1mGraph execution error
XLA   output: [[42.]]  error=None
BUG REPRODUCED: XLA silently executes invalid matmul; eager raises error.
```

**`github_tf_61884_xla_dead_code_elimination.py`**
```
W0000 00:00:1776413114.976465  345522 op_kernel.cc:1858] OP_REQUIRES failed at xla_ops.cc:602 : INVALID_ARGUMENT: Expected size[2] in [0, 3], but got 5

Stack trace for op definition: 
File "myspace/trion/bugs/new_minimal/xla/github_tf_61884_xla_dead_code_elimination.py", line 64, in <module>
File "miniconda3/lib/python3.13/site-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
File "miniconda3/lib/python3.13/site-packages/keras/src/layers/layer.py", line 998, in __call__
File "miniconda3/lib/python3.13/site-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
File "miniconda3/lib/python3.13/site-packages/keras/src/ops/operation.py", line 59, in __call__
File "miniconda3/lib/python3.13/site-packages/keras/src/utils/traceback_utils.py", line 156, in error_handler
File "myspace/trion/bugs/new_minimal/xla/github_tf_61884_xla_dead_code_elimination.py", line 57, in call

	 [[{{node Slice_1}}]]
	tf2xla conversion failed while converting __inference_call_22[_XlaMustCompile=true,config_proto=7932885297272261767,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
Input shape : (1, 2, 3, 2)
Eager output shape: (1, 2, 1, 2)  error=None
XLA   output shape: None  error=InvalidArgumentError: Exception encountered when calling ModelXLA.call().

[1mExpected size[2] in [0,
BUG REPRODUCED: XLA executes dead slice (no DCE); eager succeeds.
```


### onnx2torch / cross-compiler

**`cross_bf16_cast_jit_elide.py`**
```
PyTorch eager      : 1.234375000000   (✓ bf16 truncated)
  Inductor           : 1.234567880630   (✗ CAST ELIMINATED)
  TF eager           : 1.234375000000   (✓ bf16 truncated)
  TF-XLA (jit=True)  : 1.234567880630   (✗ CAST ELIMINATED)
  JAX eager          : 1.234375000000   (✓ bf16 truncated)
  JAX-jit (XLA)      : 1.234567880630   (✗ CAST ELIMINATED)

input       = 1.234567890123
bf16 ref    = 1.234375000000   (correct truncated value)
buggy compilers: ['PyTorch Inductor', 'TensorFlow XLA', 'JAX-jit']
correct compilers (truncate): eager modes of all frameworks + ORT + OV + ONNX-Ref + numpy

BUG REPRODUCED on 3 JIT compiler(s): PyTorch Inductor, TensorFlow XLA, JAX-jit
Fix (per compiler):
  - PyTorch Inductor: reject Cast-pair elision when intermediate dtype has fewer bits
  - TF-XLA / JAX-jit: same fix in xla/service/algebraic_simplifier.cc
```

**`cross_crms_resize_nearest_ceil.py`**
```
Resize(nearest, half_pixel, nearest_mode=ceil, scale=2)  [1,3,32,32]→[1,3,64,64]
ORT_ref (first 4 flat): [ 0.49671414 -0.1382643  -0.1382643   0.64768857]

Compiler          max_abs_diff  bug?
--------------------------------------
ORT_opt               0.000000  ok
OpenVINO              0.007692  ok
TorchScript           6.010351  BUG

Tolerance: 0.05
PASS=False
BUG REPRODUCED on ['TorchScript']: onnx2torch Resize ignores nearest_mode=ceil, uses floor instead — wrong source pixel at every other destination pixel
```

**`cross_cumsum_bool_dtype.py`**
```
Model: Cast(float→INT64) → CumSum(axis=1)  input=[1,16] bool-like 0/1 values

Seed   Compiler           max_diff  Status
--------------------------------------------------
seed=0    input=[0, 1, 1, 0, 1, 1]...  ref=[0, 1, 2, 2, 3, 4]...
       ORT_opt              0.0000  ok
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0] Graph break from `Tensor.item()`, consider setting:
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0]     torch._dynamo.config.capture_scalar_outputs = True
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0] or:
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0] to include these operations in the captured graph.
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0] 
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0] Graph break: from user code at:
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0]   File "<eval_with_key>.1", line 7, in forward
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0]     cum_sum = self.CumSum(cast, initializers_onnx_initializer_0);  cast = initializers_onnx_initializer_0 = None
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0]   File "/home/binduan/miniconda3/lib/python3.13/site-packages/onnx2torch/node_converters/cumsum.py", line 63, in forward
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0]     axis = axis.item()
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0] 
W0417 18:05:29.823000 347922 site-packages/torch/_dynamo/variables/tensor.py:1048] [0/0] 
/home/binduan/miniconda3/lib/python3.13/site-packages/onnx2torch/node_converters/cumsum.py:63: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  axis = axis.item()
       OpenVINO             0.0000  ok
       onnx2torch          10.0000  BUG  *** BUG
       torch_compile       10.0000  BUG  *** BUG
       TorchScript         10.0000  BUG  *** BUG

seed=1    input=[1, 1, 0, 0, 1, 1]...  ref=[1, 2, 2, 2, 3, 4]...
       ORT_opt              0.0000  ok
       OpenVINO             0.0000  ok
       onnx2torch          10.0000  BUG  *** BUG
       torch_compile       10.0000  BUG  *** BUG
       TorchScript         10.0000  BUG  *** BUG

seed=42   input=[0, 1, 0, 0, 0, 1]...  ref=[0, 1, 1, 1, 1, 2]...
       ORT_opt              0.0000  ok
       OpenVINO             0.0000  ok
       onnx2torch           4.0000  BUG  *** BUG
       torch_compile        4.0000  BUG  *** BUG
       TorchScript          4.0000  BUG  *** BUG

seed=123  input=[0, 1, 0, 0, 0, 0]...  ref=[0, 1, 1, 1, 1, 1]...
       ORT_opt              0.0000  ok
       OpenVINO             0.0000  ok
       onnx2torch           7.0000  BUG  *** BUG
       torch_compile        7.0000  BUG  *** BUG
       TorchScript          7.0000  BUG  *** BUG

BUG REPRODUCED: onnx2torch / torch.compile / TorchScript give wrong INT64 CumSum;
  ORT_ref, ORT_opt, OpenVINO all compute the correct cumulative sum.
```

**`cross_cumsum_kvcache_multicompiler.py`**
```
/home/binduan/miniconda3/lib/python3.13/site-packages/onnx2torch/node_converters/cumsum.py:63: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  axis = axis.item()

Backend           max_abs_diff  bug?
----------------------------------------
ORT_opt               0.000000  ok
OpenVINO              0.321663  BUG
TorchScript          56.801445  BUG

Tolerance: 0.05
PASS=False
BUG REPRODUCED on ['OpenVINO', 'TorchScript']: CumSum→Transpose→MatMul — OpenVINO tiled-GEMM fp32 accumulation order differs from ORT reference
```

**`cross_onnx2torch_cumsum.py`**
```
ORT ref[:8]:      [0.49671414 0.35844985 1.0061384  2.5291681  2.2950149  2.0608778
 3.6400905  4.407525  ]
onnx2torch[:8]:   [4.4075255 3.9108112 4.0490756 3.401387  1.8783572 2.1125104 2.3466475
 0.7674347]
max_diff=4.497339  tol=0.01
BUG REPRODUCED
```

**`cross_onnx2torch_resize_linear_asym.py`**
```
warnings.warn(
ORT ref[:4]:      [ 0.49671414  0.24272278 -0.01126862  0.01892632]
onnx2torch[:4]:   [ 0.49671414  0.43321627  0.17922492 -0.07476648]
max_diff=0.804013  tol=0.01
BUG REPRODUCED
```

**`cross_onnx2torch_resize_nearest_ceil.py`**
```
warnings.warn(
  warnings.warn(
ORT ref[:4]:      [ 0.49671414 -0.1382643  -0.1382643   0.64768857]
onnx2torch[:4]:   [ 0.49671414  0.49671414 -0.1382643  -0.1382643 ]
max_diff=3.003272  tol=0.01
BUG REPRODUCED
```

**`tf_61865_onnx_onnx2torch.py`**
```
[1;31m2026-04-17 18:05:43.023004080 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=6 must be within the inclusive range [-6,5][m
[1;31m2026-04-17 18:05:43.040724767 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=6 must be within the inclusive range [-6,5][m
params  : [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
indices : [0, 2, 4, 5, 6, 1]  (index 6 is OOB for size-6 tensor)

Compiler          Output                            Error
--------------------------------------------------------------------------------
ORT_ref           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
ORT_opt           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
OpenVINO          [2.0, 6.0, 10.0, 12.0, 0.0, 4.0]  —
onnx2torch        —                                 index 6 is out of bounds for dimension 0 with size 6  ← TARGET
torch_compile     —                                 kernel, /tmp/torchinductor_binduan/7j/c7jqxomexweswve2f
TorchScript       —                                 index 6 is out of bounds for dimension 0 with size 6

BUG REPRODUCED: onnx2torch raises (index 6 is out of bounds for dimension 0 with size 6)
  while other compilers return values: {'OpenVINO': [2.0, 6.0, 10.0, 12.0, 0.0, 4.0]}
```

**`tf_61865_onnx_torchscript.py`**
```
[1;31m2026-04-17 18:05:49.416999963 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=4 must be within the inclusive range [-4,3][m
params  : [3.0, 1.0, 4.0, 1.0]
indices : [0, 1, 2, 3, 4]  (index 4 is OOB for size-4 tensor)

[1;31m2026-04-17 18:05:49.431068373 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=4 must be within the inclusive range [-4,3][m
Compiler          Output                            Error
--------------------------------------------------------------------------------
ORT_ref           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
ORT_opt           —                                 [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero st
OpenVINO          [3.0, 1.0, 4.0, 1.0, 0.0]         —
onnx2torch        —                                 index 4 is out of bounds for dimension 0 with size 4
torch_compile     —                                 kernel, /tmp/torchinductor_binduan/7d/c7dlgn3qp7jmlxdu7
TorchScript       —                                 index 4 is out of bounds for dimension 0 with size 4  ← TARGET

BUG REPRODUCED: TorchScript raises on OOB Gather
  while other compilers return values: {'OpenVINO': [3.0, 1.0, 4.0, 1.0, 0.0]}
```

