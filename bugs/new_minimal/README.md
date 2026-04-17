# ML Compiler Bug Collection — Minimal Reproducible Scripts

**122 verified bugs** across 7 ML compilers/runtimes.

All scripts:
- Build their ONNX model from scratch with `onnx.helper` (no binary blobs)
- Print correct (ORT) vs buggy output side-by-side
- Exit 0 = BUG REPRODUCED, 1 = not reproduced, 2 = missing dep

Verified on: Python 3.13, ONNX 1.21, ONNXRuntime 1.24.4, OpenVINO 2026.0,
TensorFlow 2.21.0, PyTorch 2.9.1, TVM master, onnx2torch latest.

---

## Summary

| Compiler | Bugs | Root Cause |
|----------|------|------------|
| OpenVINO | 58 | fp32 MatMul/Conv tiling precision; uint8/int8 saturation; NaN propagation; GPU tile overflow; OOB Gather silent return |
| torch.compile | 27 | `Reshape` runtime shape → `InternalTorchDynamoError`; index_add shape bypass; transpose+reduce fusion stride bug; bitshift UB |
| ONNXRuntime | 9 | Resize nearest/bilinear/cubic; int-truncation fusion; SIGFPE mod-zero; GridSample; BitShift shift-by-64 |
| onnx2torch | 9 | CumSum graph break; resize coord mode; nearest-ceil; INT64 cumsum; bf16 cast elision; OOB Gather; TorchScript resize |
| XLA / TensorFlow | 8 | OOB gather silent clamp; autocluster DCE; dead-code slice; matmul incompatible shapes; fp32 GEMM accumulation |
| TFLite | 7 | FC fusion converter crash; L2-norm axis semantics; unstack-concat fold; Tanh precision; GEMM accumulation |
| TVM | 4 | SimplifyExpr rsqrt; LiftTransformParams off-by-one; GELU approx; cubic resize |

---

## Directories

- `onnxruntime/` — ONNXRuntime bugs (9 scripts)
- `openvino/` — OpenVINO CPU bugs (58 scripts)
- `tflite/` — TFLite converter/runtime bugs (7 scripts)
- `torch_compile/` — PyTorch torch.compile / Inductor bugs (27 scripts)
- `tvm/` — Apache TVM compiler bugs (4 scripts)
- `xla/` — TensorFlow/XLA bugs (8 scripts)
- `onnx2torch/` — onnx2torch conversion bugs and cross-compiler multi-backend bugs (9 scripts)

---

## Running

```bash
# Run a single script
python3 openvino/cross_openvino_reciprocal_mul.py

# Run all scripts for one compiler and report pass/fail
for f in openvino/*.py; do
    python3 "$f" > /dev/null 2>&1 && echo "BUG $f" || echo "SKIP $f"
done

# Run all 122 scripts with timing
for d in onnxruntime openvino tflite torch_compile tvm xla onnx2torch; do
    echo "=== $d ==="
    for f in "$d"/*.py; do
        out=$(timeout 60 python3 "$f" 2>/dev/null | tail -1)
        echo "  exit=$?  $(basename $f)  $out"
    done
done
```

Exit codes: **0** = bug reproduced, **1** = not reproduced (fixed), **2** = required dependency missing.

---

## ONNXRuntime (9 bugs)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `github_onnx_spec_007.py` | `BUG REPRODUCED` | wrong index at elem 4 | Resize nearest `half_pixel`+`round_prefer_ceil` 20→6: ORT returns 0.7368 at element 4 (should be 0.7895) |
| `github_ort_002.py` | `BUG REPRODUCED` | 12/64 mismatched | Nearest resize 26→64 (scale 64/26): one-pixel-off vs PyTorch |
| `github_ort_003.py` | `BUG REPRODUCED` | max abs 0.2947 | Asymmetric bilinear resize max error vs PyTorch |
| `github_ort_004.py` | `BUG REPRODUCED` | all True | `ORT_ENABLE_ALL` optimizer fuses `float→int32→bool` to `float→bool`, skipping required int-truncation |
| `github_ort_008.py` | `BUG REPRODUCED` | max abs 0.061 | CPU cubic resize (`pytorch_half_pixel`, `antialias=0`) diverges from PyTorch bicubic |
| `github_ort_016.py` | `BUG REPRODUCED` | out of range | `GridSample(bicubic, padding=border)` clamps after neighbourhood lookup instead of per-sample |
| `github_ort_017_mod_int_divzero_sigfpe.py` | `BUG REPRODUCED — ORT Mod(fmod=0, int32) with B=0 triggers SIGFPE` | SIGFPE (signal 8) | `Mod(fmod=0, int32)` with zero divisor: raw `a % b` without guard triggers hardware SIGFPE |
| `cross_bitshift_shift64_ov_ort.py` | `BUG REPRODUCED on ['ORT_noopt', 'ORT_opt', 'ORT_noopt_LEFT', 'OpenVINO_LEFT']: BitShift by 64 returns non-zero (C UB — x86 masks shift to 6 bits)` | shift→64 non-zero | BitShift RIGHT by 64: ORT (and OpenVINO) hit C UB; x86 masks shift count to 6 bits so `n=64` acts as `n=0` |
| `tf_61865_onnx_ort.py` | `BUG REPRODUCED: ORT (ref + opt) raise INVALID_ARGUMENT on OOB Gather` | INVALID_ARGUMENT | `Gather(params[5], oob_idx=5)`: both ORT ref and opt raise `INVALID_ARGUMENT`; OpenVINO silently returns 0.0 |

---

## OpenVINO (58 bugs)

### Integer arithmetic saturation (ONNX requires modular wrap)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `github_ov_019_uint8_sub_no_wrap.py` | `BUG REPRODUCED — OV uint8 Sub saturates instead of wrapping` | `251 → 0` | uint8 Sub saturates instead of modular wrapping (ONNX spec: mod 256) |
| `github_ov_020_uint8_mul_no_wrap.py` | `BUG REPRODUCED — OV uint8 Mul saturates instead of wrapping` | `40000 → 255` | uint8 Mul saturates instead of wrapping |
| `github_ov_021_uint8_add_no_wrap.py` | `BUG REPRODUCED — OV uint8 Add saturates instead of wrapping` | `300 → 255` | uint8 Add saturates instead of wrapping |
| `github_ov_024_int8_sub_saturation.py` | `BUG REPRODUCED — OV int8 Sub saturates instead of two's-complement wrapping` | `-128−1 → -128` | int8 Sub saturates instead of two's-complement wrapping |
| `github_ov_025_int8_add_saturation.py` | `BUG REPRODUCED — OV int8 Add saturates instead of two's-complement wrapping` | `100+100 → 127` | int8 Add saturates instead of two's-complement wrapping |

### NaN/Inf propagation (IEEE 754)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `github_ov_023_relu_nan_propagation.py` | `BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754)` | NaN → 0 | `Relu(NaN)` returns 0.0; should propagate NaN per IEEE 754 |
| `github_ov_026_exp_nan_to_inf.py` | `BUG REPRODUCED — OV Exp(NaN) → +inf, should propagate NaN (IEEE 754)` | NaN → +inf | `Exp(NaN)` returns +inf; should propagate NaN |
| `github_ov_022_reducelogsumexp_overflow.py` | `BUG REPRODUCED — OV ReduceLogSumExp overflows for inputs ≥ 88.7 (fp32 exp limit)` | returns `inf` | `ReduceLogSumExp` not implemented with max-subtraction trick; `exp(100)` overflows fp32 |

### GPU tile overflow

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `github_ov_015_matmul_gpu_tile_overflow.py` | `BUG REPRODUCED: MatMul GPU tile skips partial last tile for dim > 2048` | dim>2048 returns 2048 | GPU MatMul kernel used fixed tile size 2048; partial last tile skipped for inner-dim > 2048 |

### fp32 GEMM/Conv tiling precision

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `cross_openvino_conv_bn_fusion.py` | `BUG REPRODUCED` | max abs 0.022 | Conv+BN fusion rounding error exceeds 0.01 tolerance vs ORT unoptimised |
| `cross_openvino_conv_fp32_precision.py` | `BUG REPRODUCED: OpenVINO CPU Conv fp32 accumulation differs from ORT reference` | max abs 0.054 | Conv picks Winograd/tiled GEMM; fp32 accumulation differs from ORT (5×5: 0.083) |
| `cross_openvino_fp16_matmul_add.py` | `BUG REPRODUCED: OpenVINO fp16 GEMM tile accumulation diverges from ORT` | max abs 0.078 | fp16 tiled GEMM accumulates partial sums in different order than ORT |
| `cross_openvino_add_relu_sub.py` | `BUG REPRODUCED: OpenVINO add_relu_sub (max_abs=0.1404)` | max abs >0.01 | MatMul→Add→Relu→Sub: OV CPU GEMM fp32 inner-dimension accumulation differs from ORT |
| `cross_openvino_add_self_sub_double.py` | `BUG REPRODUCED: OpenVINO add_self_sub_double (max_abs=0.1797)` | max abs >0.01 | MatMul→Add(R)→Add(R)→Sub(R): double add/sub residual exposes OV GEMM fp32 rounding |
| `cross_openvino_alibi_attention.py` | `BUG REPRODUCED: OpenVINO alibi_attention (max_abs=0.1724)` | max abs >0.01 | ALiBi attention MatMul(Q,Kt)→Softmax→MatMul(xV): OV GEMM fp32 accumulation in large inner-dim |
| `cross_openvino_aspp_dilated_branch.py` | `BUG REPRODUCED: OpenVINO aspp_dilated_branch (max_abs=0.2501)` | max abs >0.01 | ASPP Conv(3×3,dil=2)+Conv(3×3,dil=4)→Add: OV Conv large C_IN fp32 tiling |
| `cross_openvino_attention_causal_only.py` | `BUG REPRODUCED: OpenVINO attention_causal_only (max_abs=0.2295)` | max abs >0.01 | Causal attention MatMul+mask+Softmax: OV GEMM fp32 accumulation amplified by Softmax |
| `cross_openvino_attention_logit_softcap.py` | `BUG REPRODUCED: OpenVINO attention_logit_softcap (max_abs=38.6729)` | max abs >0.01 | Attention softcap MatMul→Div→Tanh→Mul: OV GEMM fp32 diff preserved through nonlinearity |
| `cross_openvino_attention_with_sink_token.py` | `BUG REPRODUCED: OpenVINO attention_with_sink_token (max_abs=0.1724)` | max abs >0.01 | Attention with sink bias MatMul+bias+Softmax: OV GEMM fp32 diff amplified by sink token |
| `cross_openvino_broadcast_1d_scalar.py` | `BUG REPRODUCED: OpenVINO broadcast_1d_scalar (max_abs=0.1404)` | max abs >0.01 | MatMul→Add(1D scalar broadcast bias): OV GEMM fp32 accumulation with 1D bias broadcast |
| `cross_openvino_conv_add_relu.py` | `BUG REPRODUCED: OpenVINO conv_add_relu (max_abs=0.1462)` | max abs >0.01 | Conv(3×3)→Add→Relu: OV CPU Conv fp32 inner-dim sum tiling differs from ORT |
| `cross_openvino_conv_bn_eval_explicit.py` | `BUG REPRODUCED: OpenVINO conv_bn_eval_explicit (max_abs=0.7198)` | max abs >0.01 | Conv(3×3)→BatchNorm (eval mode): OV Conv+BN fusion fp32 rounding differs from ORT |
| `cross_openvino_conv_bn_relu6.py` | `BUG REPRODUCED: OpenVINO conv_bn_relu6 (max_abs=0.3779)` | max abs >0.01 | Conv→BatchNorm→Clip(0,6) [ReLU6]: OV Conv+BN fusion fp32 rounding; clamp amplifies diff |
| `cross_openvino_conv_prelu_channel.py` | `BUG REPRODUCED: OpenVINO conv_prelu_channel (max_abs=0.1462)` | max abs >0.01 | Conv(3×3)→PRelu(per-channel): OV Conv fp32 tiling; PRelu channels expose error |
| `cross_openvino_ei_log_zero.py` | `BUG REPRODUCED: OpenVINO ei_log_zero (max_abs=0.2028)` | max abs >0.01 | MatMul→Log: OV GEMM fp32 diff causes different inputs to Log near zero |
| `cross_openvino_einsum_transpose.py` | `BUG REPRODUCED: OpenVINO einsum_transpose (max_abs=0.5448)` | max abs >0.01 | MatMul→Transpose→MatMul (einsum-style): OV GEMM fp32 inner-dim accumulation |
| `cross_openvino_expm1_bounded.py` | `BUG REPRODUCED: OpenVINO expm1_bounded (max_abs=0.1484)` | max abs >0.01 | Exp(x)→MatMul wide inner-dim: OV GEMM tiles in different order than ORT |
| `cross_openvino_flatten_gemm.py` | `BUG REPRODUCED: OpenVINO flatten_gemm (max_abs=0.1540)` | max abs >0.01 | Flatten→Gemm(W,b): OV CPU GEMM fp32 inner-dim accumulation differs from ORT |
| `cross_openvino_flex_attention_precision_sdpa.py` | `BUG REPRODUCED: OpenVINO flex_attention_precision_sdpa (max_abs=0.1724)` | max abs >0.01 | SDPA: MatMul(Q,Kt)×scale→Softmax→MatMul(xV): OV GEMM fp32 diff in attention path |
| `cross_openvino_gemm_sigmoid.py` | `BUG REPRODUCED: OpenVINO gemm_sigmoid (max_abs=0.0250)` | max abs >0.01 | Gemm(x,W,b)→Sigmoid: OV CPU GEMM fp32 inner-dim accumulation differs from ORT |
| `cross_openvino_global_branch_mul.py` | `BUG REPRODUCED: OpenVINO global_branch_mul (max_abs=5.3273)` | max abs >0.01 | Two parallel Conv paths multiplied: OV applies different fp32 GEMM tiling per branch |
| `cross_openvino_glu.py` | `BUG REPRODUCED: OpenVINO glu (max_abs=0.5281)` | max abs >0.01 | MatMul→Split→Sigmoid(gate)→Mul [GLU]: OV CPU GEMM fp32 inner-dim accumulation |
| `cross_openvino_group_query_attention.py` | `BUG REPRODUCED: OpenVINO group_query_attention (max_abs=0.0550)` | max abs >0.01 | MatMul+Reshape (multi-head format): OV tiles differently; error accumulates |
| `cross_openvino_ia_sat_sub_uint8_underflow.py` | `BUG REPRODUCED: OpenVINO ia_sat_sub_uint8 (MatMul pattern, max_abs=0.1404)` | max abs >0.01 | Cast(float→uint8)→Clip simulating uint8 subtraction underflow: OV integer boundary differs |
| `cross_openvino_inception_v3_branch.py` | `BUG REPRODUCED: OpenVINO inception_v3_branch (max_abs=0.3883)` | max abs >0.01 | Inception: Conv(1×1,3×3,5×5)→Concat: OV Conv fp32 tiling multi-branch accumulation |
| `cross_openvino_linear_relu_chain.py` | `BUG REPRODUCED: OpenVINO Gemm+Relu chain produces wrong fp32 output` | max diff 1.2–3.5 | Three consecutive Gemm(transB=1)→Relu: OV Gemm+Relu chain fusion wrong fp32 output |
| `cross_openvino_matmul_add_biasgelu_bcast.py` | `BUG REPRODUCED: OpenVINO matmul_add_biasgelu_bcast (max_abs=0.1941)` | max abs >0.01 | MatMul→Add(bias)→GELU→Mul(gate): OV GEMM fp32 diff; GELU nonlinearity amplifies |
| `cross_openvino_matmul_add_layernorm.py` | `BUG REPRODUCED: OpenVINO matmul_add_layernorm (max_abs=0.0086)` | max abs >0.01 | MatMul→Add→LayerNorm: OV GEMM inner-dim fp32 rounding; LayerNorm amplifies |
| `cross_openvino_maxpool_bad_pad.py` | `BUG REPRODUCED: OpenVINO silently accepts MaxPool with pad >= kernel` | silent accept | `MaxPool(kernel=k, pad≥k)` violates ONNX spec: ORT rejects; OV silently computes extended output shape |
| `cross_openvino_multi_query_attention.py` | `BUG REPRODUCED: OpenVINO multi_query_attention (max_abs=0.1724)` | max abs >0.01 | Multi-query attention: Q from large MatMul, shared K/V: OV GEMM fp32 diff in Q projection |
| `cross_openvino_multi_scale_conv_branch.py` | `BUG REPRODUCED: OpenVINO multi_scale_conv_branch (max_abs=0.2586)` | max abs >0.01 | Multi-scale Conv branches→Add: OV Conv fp32 inner-dim accumulation differs from ORT |
| `cross_openvino_neg_unary.py` | `BUG REPRODUCED: OpenVINO neg_unary (max_abs=0.1747)` | max abs >0.01 | Conv(3×3)→Neg: OV Conv fp32 tiling; negation exposes sign difference |
| `cross_openvino_pad_conv.py` | `BUG REPRODUCED: OpenVINO pad_conv (max_abs=0.1808)` | max abs >0.01 | Pad(reflect)→Conv(3×3): OV Conv with reflect padding fp32 tiling differs from ORT |
| `cross_openvino_pointwise_dw_block.py` | `BUG REPRODUCED: OpenVINO pointwise_dw_block (max_abs=2.1500)` | max abs >0.01 | Conv(1×1 pointwise)→Conv(3×3 depthwise)→Conv(1×1): OV depthwise Conv fp32 tiling |
| `cross_openvino_reciprocal_mul.py` | `BUG REPRODUCED: OpenVINO reciprocal_mul (max_abs=41.4429)` | max abs >0.01 | Reciprocal(x)→Mul(W) with small values: OV fp32 precision differs for large reciprocals |
| `cross_openvino_reduce_l2_last.py` | `BUG REPRODUCED: OpenVINO reduce_l2_last (max_abs=0.0718)` | max abs >0.01 | MatMul→ReduceL2(last axis): OV GEMM fp32 diff; ReduceL2 amplifies in large dims |
| `cross_openvino_redundant_reshape.py` | `BUG REPRODUCED: OpenVINO redundant_reshape (max_abs=0.1404)` | max abs >0.01 | MatMul→Reshape→Reshape (redundant): OV may eliminate incorrectly or GEMM fp32 accumulation |
| `cross_openvino_rrelu_inference_identity.py` | `BUG REPRODUCED: OpenVINO rrelu_inference_identity (max_abs=0.1230)` | max abs >0.01 | RReLU→PReLU in inference: OV applies Winograd/tiled GEMM then fuses PReLU, reordering fp32 |
| `cross_openvino_slice_full_range.py` | `BUG REPRODUCED: OpenVINO slice_full_range (max_abs=0.2065)` | max abs >0.01 | MatMul→Slice (full range): OV CPU GEMM fp32 diff; full-range Slice preserves it |
| `cross_openvino_space_to_depth_block.py` | `BUG REPRODUCED: OpenVINO space_to_depth_block (max_abs=1.0417)` | max abs >0.01 | Conv(3×3)→SpaceToDepth→Conv(1×1): OV Conv fp32 tiling; SpaceToDepth reorders channels |
| `cross_openvino_sub_self_mul_zero.py` | `BUG REPRODUCED: OpenVINO sub_self_mul_zero (max_abs=2.9208)` | max abs >0.01 | MatMul→Sub(mm,mm)×const [should be near-zero]: OV fp32 rounding in Sub(mm,mm) |
| `cross_openvino_tile_conv.py` | `BUG REPRODUCED: OpenVINO tile_conv (max_abs=0.1471)` | max abs >0.01 | Tile(x)→Conv(3×3): OV Conv with tiled input fp32 tiling differs from ORT |
| `cross_openvino_transformer_encoder_layer.py` | `BUG REPRODUCED: OpenVINO transformer_encoder_layer (max_abs=0.0135)` | max abs >0.01 | Full transformer: self-attn+FFN with GELU: OV GEMM fp32 diff in transformer |
| `cross_openvino_transpose_matmul_transpose.py` | `BUG REPRODUCED: OpenVINO transpose_matmul_transpose (max_abs=0.1714)` | max abs >0.01 | Transpose→MatMul→Transpose: OV GEMM fp32 accumulation after transpose reorder |
| `cross_openvino_transpose_transpose_squash.py` | `BUG REPRODUCED: OpenVINO transpose_transpose_squash (max_abs=0.0890)` | max abs >0.01 | Transpose([0,2,3,1])→Transpose([0,3,1,2])→MatMul: OV optimizer may squash to wrong identity |
| `cross_openvino_triple_add_residual.py` | `BUG REPRODUCED: OpenVINO triple_add_residual (max_abs=0.1797)` | max abs >0.01 | MatMul→Add(r1)→Add(r2)→Add(r3) [triple residual]: OV GEMM fp32 diff |
| `cross_openvino_where_mask_fill.py` | `BUG REPRODUCED: OpenVINO where_mask_fill (max_abs=0.1797)` | max abs >0.01 | MatMul→Where(mask, result, fill): OV GEMM fp32 diff; Where exposes |
| `tf_61865_onnx_openvino.py` | `ORT_ref/ORT_opt raise INVALID_ARGUMENT; OpenVINO proceeds silently.` | OOB → 0.0 silent | `Gather(params[12], oob_idx=12)`: OpenVINO silently returns 0.0 for OOB slot; ORT raises `INVALID_ARGUMENT` |

---

## TFLite (7 bugs)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `cross_tflite_unstack_concat_reshape_fold.py` | `TFLite [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] WRONG` | output = input | TFLite converter folds `reshape→unstack→concat→reshape` into a no-op RESHAPE, dropping the transpose-equivalent element permutation |
| `cross_tflite_l2_normalize_multi_shape.py` | `TFLite [1.0, 1.0, 1.0, 1.0] WRONG` | per-row vs whole-tensor | TFLite `L2_NORMALIZATION` normalizes only the innermost dim instead of whole tensor (axis=None); tested on rank-3 `(2,2,3)` and `(4,1)` |
| `cross_tflite_fully_connected_mul_fusion_crash.py` | `TFLite=CONVERTER CRASH: <unknown>:0: error: loc(fused[...` | converter crash | `multiply(x, scalar)→matmul(W)→add(bias)` triggers crash in `Mul+MatMul+Add → tfl.fully_connected` fusion: infers input as `tensor<1xf32>` instead of `tensor<1xNxf32>` |
| `cross_tflite_attention_logit_softcap.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` | max abs >0.01 | TFLite Tanh fp32 precision differs from ORT at moderate inputs in `Div→Tanh→Mul` (attention softcap) |
| `cross_tflite_sub_self_mul_zero.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` | max abs >1e-4 | TFLite `Sub(x,x)→MatMul` may not preserve exact zero for sub-self pattern |
| `cross_tflite_transformer_encoder_layer.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` | max abs >0.01 | TFLite fp32 GEMM accumulation differs from ORT in full self-attention+FFN transformer |
| `cross_tflite_transpose_matmul_transpose.py` | `BUG REPRODUCED (TFLite error): AttributeError: '_SignatureMap' object has no attribute 'name'` | max abs >0.01 | TFLite GEMM accumulation differs from ORT after Transpose→MatMul→Transpose |

---

## torch.compile (27 bugs)

### MatMul→Reshape runtime shape crashes (onnx2torch + Inductor)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `cross_torch_compile_add_self_sub_double.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | MatMul→Reshape: onnx2torch Reshape with runtime shape tensor; Inductor cannot compile |
| `cross_torch_compile_aspp_dilated_branch.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on ASPP dilated branch pattern |
| `cross_torch_compile_conv_k7_stride2.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on Conv k=7 stride=2 pattern |
| `cross_torch_compile_conv_manual_celu.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on Conv+manual CELU pattern |
| `cross_torch_compile_cp_fp16_matmul_n512.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on fp16 MatMul N=512 pattern |
| `cross_torch_compile_einsum_transpose.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on einsum-style Transpose pattern |
| `cross_torch_compile_flatten_gemm.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on Flatten→Gemm pattern |
| `cross_torch_compile_gemm_sigmoid.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on Gemm→Sigmoid pattern |
| `cross_torch_compile_group_query_attention.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on group-query attention |
| `cross_torch_compile_ia_sat_sub_uint8_underflow.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on uint8 subtraction underflow pattern |
| `cross_torch_compile_manual_hard_swish.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on manual hard-swish pattern |
| `cross_torch_compile_matmul_add_layernorm.py` | `BUG REPRODUCED: torch.compile crashes on matmul_add_layernorm` | InternalTorchDynamoError | Same root cause on MatMul+Add+LayerNorm pattern |
| `cross_torch_compile_multi_query_attention.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on multi-query attention |
| `cross_torch_compile_reciprocal_mul.py` | `BUG REPRODUCED: torch.compile crashes on reciprocal_mul` | InternalTorchDynamoError | Reciprocal→Reshape→MatMul: runtime shape tensor after Reciprocal |
| `cross_torch_compile_reduce_l2_last.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on ReduceL2 last-axis pattern |
| `cross_torch_compile_redundant_reshape.py` | `BUG REPRODUCED: torch.compile crashes on redundant_reshape: InternalTorchDynamoError` | InternalTorchDynamoError | MatMul+double Reshape: dynamic shape in Inductor fails |
| `cross_torch_compile_slice_full_range.py` | `BUG REPRODUCED: torch.compile crashes on slice_full_range` | InternalTorchDynamoError | Same root cause on Slice full-range pattern |
| `cross_torch_compile_space_to_depth_block.py` | `BUG REPRODUCED: torch.compile crashes on space_to_depth_block: NotImplementedError: Converter is not implemented` | InternalTorchDynamoError | SpaceToDepth block: onnx2torch converts shape; Inductor shape propagation fails |
| `cross_torch_compile_sub_self_mul_zero.py` | `BUG REPRODUCED: torch.compile crashes on sub_self_mul_zero` | InternalTorchDynamoError | Sub(x,x)→Reshape→MatMul: runtime shape tensor after Sub |
| `cross_torch_compile_topk_last_axis_k1.py` | `BUG REPRODUCED: torch.compile crashes on topk_last_axis_k1` | InternalTorchDynamoError | TopK→Reshape: runtime shape tensor after TopK |
| `cross_torch_compile_transpose_transpose_squash.py` | `BUG REPRODUCED: torch.compile crashes on transpose_transpose_squash: InternalTorchDynamoError` | InternalTorchDynamoError | Double Transpose: Inductor shape propagation fails |
| `cross_torch_compile_triple_add_residual.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on triple residual-add pattern |
| `cross_torch_compile_where_mask_fill.py` | `ORT Reshape error: input_shape_size==size was false (Input:{4,32}, requested:{4,64})` | InternalTorchDynamoError | Same root cause on Where+mask+fill pattern |

### Other torch.compile bugs

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `github_inductor_009_transpose_reduce_fusion.py` | `BUG REPRODUCED: Inductor transpose+reduce fusion reinterprets storage strides` | max abs 1.53e-5 | Inductor fuses `ReduceSum`+`Transpose().contiguous()` and reinterprets storage strides; wrong output in attention-grad pattern |
| `github_inductor_011_bitshift_ub_shift64.py` | `BUG REPRODUCED: BitShift by 64 returns non-zero (C UB: x86 masks shift to 6 bits)` | shift→64 non-zero | Inductor CPU C++ codegen emits native `x >> n` which is UB when `n == word_width`; x86 masks to 6 bits so `n=64` acts as `n=0` |
| `pt_121135_torch_compile.py` | `BUG REPRODUCED: torch.compile skips shape validation in index_add` | silent pass | `x.index_add(0, randperm_index, src)` where `src` has incompatible trailing dims: eager/TorchScript raise `RuntimeError`; `torch.compile` skips shape validation silently |
| `tf_61865_onnx_torch_compile.py` | `BUG REPRODUCED: torch.compile raises on OOB Gather` | runtime error | `Gather(params[10], oob_idx=10)`: torch.compile generates C++ kernel that runtime-errors on OOB index; OpenVINO silently returns 0.0 |

---

## TVM (4 bugs)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `github_tvm_004.py` | `BUG REPRODUCED` | max abs 0.053 | `Resize(cubic, half_pixel, coeff_a=-0.5)` diverges from ORT and PyTorch bicubic reference |
| `github_tvm_010_simplifyexpr_rsqrt_precision.py` | `BUG REPRODUCED: SimplifyExpr rsqrt rewrite introduces large precision error` | rel_err 9983× | `SimplifyExpr` rewrites `sqrt(x)/y → rsqrt(x)*y`; fast-rsqrt (bit-trick) has catastrophic error for small `x`; also `FoldScaleAxis` reorders `conv→relu→mul` incorrectly for negative scales |
| `github_tvm_011_lifttransformparams_const_bind.py` | `BUG REPRODUCED: LiftTransformParams corrupts constant binding` | off-by-one | `LiftTransformParams` re-binds shared `ones` constant to wrong lifted slot; `(A+1)*(B+1)` computes as `(A+2)*(B+2)`, max diff 33 |
| `github_tvm_012_gelu_approx_tanh.py` | `BUG REPRODUCED: TVM maps ONNX Gelu(approx=tanh) → exact erf, producing systematic error > 1e-4` | max abs 1.53e-4 | TVM maps ONNX `Gelu(approximate='tanh')` to exact erf-based GELU, producing systematic ~1.5e-4 error on the Transformer GELU fast-path |

---

## XLA / TensorFlow (8 bugs)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `github_tensorflow_002.py` | `BUG REPRODUCED` | off-by-one row | MLIR/TOSA nearest resize shifts rows by 1 with `half_pixel_centers=True` ([TF #62386](https://github.com/tensorflow/tensorflow/issues/62386)) |
| `github_tf_61881_xla_matmul_incompatible_shapes.py` | `BUG REPRODUCED: XLA silently executes invalid matmul; eager raises error.` | silent wrong result | `tf.matmul([1,4], [6,1])` — inner dims 4≠6: TF eager raises `InvalidArgumentError`; XLA silently executes and returns `[[42.]]` |
| `github_tf_61884_xla_dead_code_elimination.py` | `BUG REPRODUCED: XLA executes dead slice (no DCE); eager succeeds.` | crash instead of DCE | Dead `tf.slice` with OOB `size[2]`: TF eager skips (no error); XLA fails to DCE and executes it, raising `InvalidArgumentError` |
| `cross_xla_gather_oob_clamp_variants.py` | `-> BUG: XLA silently produced output instead of erroring` | silent clamp 8/8 | XLA `Gather` silently clamps OOB indices to `[0, dim-1]` in 8 variants (off-by-one, INT_MAX, negative, 2D axis=0/1); TF eager/PyTorch raise in every case |
| `cross_xla_autocluster_slice_oob_suppress.py` | `V4: shape=(3,2,3,4) RAISED InvalidArgumentError RAISED InvalidArgumentError OK [4.0, 5.0, 6.0]... <- BUG` | OOB silently ignored | `tf_xla_auto_jit=2` dead-code-eliminates a Slice with OOB `size[2]` before bounds-checking; XLA jit and TF eager both raise `InvalidArgumentError` |
| `cross_xla_add_self_sub_double.py` | `BUG REPRODUCED: XLA add_self_sub_double (max_abs=0.0203)` | max abs >0.01 | XLA fp32 GEMM accumulation / optimization differs from ORT on MatMul→Add(R)→Add(R)→Sub(R) |
| `cross_xla_matmul_add_biasgelu_bcast.py` | `BUG REPRODUCED: XLA matmul_add_biasgelu_bcast (max_abs=0.0678)` | max abs >0.01 | XLA fp32 GEMM+GELU accumulation differs from ORT on MatMul→Add(bias)→GELU→Mul(gate) |
| `cross_xla_space_to_depth_block.py` | `BUG REPRODUCED: XLA space_to_depth_block (max_abs=0.1047)` | max abs >0.01 | XLA Conv+SpaceToDepth fp32 accumulation differs from ORT |

---

## onnx2torch (9 bugs)

| Script | Output | Error | Root Cause |
|--------|--------|-------|------------|
| `cross_onnx2torch_cumsum.py` | `BUG REPRODUCED` | max diff 4.50 | `CumSum(axis=2)`: onnx2torch calls `axis.item()`, triggering a `torch.compile` graph break; broken graph produces wrong cumulative sum |
| `cross_onnx2torch_resize_linear_asym.py` | `BUG REPRODUCED` | max diff 0.80 | onnx2torch maps ONNX `asymmetric` coordinate_transformation_mode to wrong PyTorch interpolation mode |
| `cross_onnx2torch_resize_nearest_ceil.py` | `BUG REPRODUCED` | max diff 3.00 | onnx2torch converts `nearest_mode='ceil'` to `F.interpolate`, which only supports floor nearest |
| `cross_cumsum_bool_dtype.py` | `BUG REPRODUCED: onnx2torch / torch.compile / TorchScript give wrong INT64 CumSum` | max diff 4–10 | `Cast(float→INT64)→CumSum(axis=1)` with 0/1 inputs: onnx2torch/torch.compile/TorchScript return wrong INT64 cumulative sums; distinct from graph-break bug |
| `cross_bf16_cast_jit_elide.py` | `- TF-XLA / JAX-jit: same fix in xla/service/algebraic_simplifier.cc` | 3 compilers fail | Unified repro: Inductor, TF-XLA, and JAX-jit all eliminate the bf16↔fp32 cast pair; eager modes of all three plus ORT/OV/numpy/ONNX-Ref are correct |
| `cross_crms_resize_nearest_ceil.py` | `BUG REPRODUCED on ['TorchScript']: onnx2torch Resize ignores nearest_mode=ceil, uses floor instead` | TorchScript off by 6.0 | `Resize(nearest, nearest_mode=ceil)`: onnx2torch silently maps to PyTorch's floor; ORT and OV agree |
| `cross_cumsum_kvcache_multicompiler.py` | `BUG REPRODUCED on ['OpenVINO', 'TorchScript']: CumSum→Transpose→MatMul — OpenVINO tiled-GEMM fp32 accumulation order differs from ORT reference` | OV diff 0.32, TorchScript diff 57 | `CumSum→Transpose→MatMul` (Q×Kᵀ self-attn): OV tiled GEMM error; TorchScript onnx2torch wrong value |
| `tf_61865_onnx_onnx2torch.py` | `BUG REPRODUCED: onnx2torch raises (index 6 is out of bounds for dimension 0 with size 6)` | IndexError | `Gather(params[6], oob_idx=6)`: onnx2torch raises `IndexError` for OOB slot; OpenVINO silently returns 0.0 |
| `tf_61865_onnx_torchscript.py` | `BUG REPRODUCED: TorchScript raises on OOB Gather` | IndexError | `Gather(params[4], oob_idx=4)`: TorchScript raises `IndexError` for OOB slot; OpenVINO silently returns 0.0 |

---

## Environment requirements

| Package | Min version | Used by |
|---------|-------------|---------|
| `onnx` | 1.21 | all scripts |
| `onnxruntime` | 1.24.4 | all scripts (reference oracle) |
| `numpy` | any | all scripts |
| `openvino` | 2026.0 | `openvino/`, `cross_bitshift_*`, `tf_61865_onnx_openvino` |
| `tensorflow` | 2.21.0 | `xla/`, `tflite/` |
| `torch` | 2.9.1 | `torch_compile/`, `onnx2torch/` |
| `onnx2torch` | latest | `torch_compile/cross_*`, `onnx2torch/` |
| `tvm` | master | `tvm/` (analytical if not installed) |

Scripts exit 2 if the required backend is missing — they do not crash.
