# Root Cause Analysis — 123 Bugs

## OnnxRuntime (60 bugs)

### Resize operator (16 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_ort_001 | Resize(nearest, tf_half_pixel_for_nn, round_prefer_ceil, batch=2) | tf_half_pixel_for_nn applied to batch/channel dims; batch1 returns batch0 values |
| github_ort_002 | Resize(nearest, half_pixel, round_prefer_ceil, scale=64/26) | Pixel selection off by one vs PyTorch reference |
| github_ort_003 | Resize(linear, asymmetric, scale=2.5) | Asymmetric coordinate bilinear gives up to 0.29 error vs PyTorch |
| github_ort_008 | Resize(cubic, pytorch_half_pixel, antialias=0, downscale 8->4) | CUDA cubic resize has wrong antialias grid setup and hardcoded cubic_coeff_a |
| github_ort_010 | Reshape->MatMul->Reshape->Resize(linear) on GPU | GPU Resize linear outputs constant 0.5 when preceded by MatMul |
| github_onnx_spec_002 | Resize(scale=1.5, input_size=7, 7*1.5=10.5) | Spec ambiguous on floor vs round for fractional output dims; ORT uses floor=10 |
| github_onnx_spec_003 | Resize(pytorch_half_pixel, output_size=1, width=4->1) | Formula gives x_orig=-0.5; ORT clips to 0, returns first element instead of center |
| github_onnx_spec_007 | Resize(nearest, half_pixel, round_prefer_ceil, scale 20->6) | Wrong index for element 4 due to spec ambiguity in half_pixel rounding |
| github_onnx_spec_13 | Resize opset-12->13 adapter, tf_half_pixel_for_nn | CompatibleAdapter mishandled deprecated tf_half_pixel_for_nn attribute |
| campaign_v2_0002 | Resize(nearest, ceil, half_pixel, scale 2x) + Cast | Nearest ceil + half_pixel selects different boundary pixels than floor mode |
| campaign_v2_0024 | Resize(linear, half_pixel) + Resize(nearest_ceil, half_pixel) in sequence | ORT optimizer incorrectly merges/reorders two consecutive Resize ops with different modes |
| campaign_v2_0028 | Resize(nearest_ceil, half_pixel, 2x) + ReduceSum(axis=2) | ORT fuses Resize with downstream ReduceSum, breaks spatial dimension tracking |
| campaign_v2_0029 | Resize(nearest_ceil, half_pixel) before dilated ASPP Conv | ORT reorders Resize with downstream dilated convolutions |
| campaign_v2_0053 | Abs(x)*x + Resize(nearest_ceil, half_pixel, 2x) + MatMul | Nearest_ceil boundary index computation wrong after signed-square |
| campaign_v2_0055 | LayerNorm + ASPP + ConstantOfShape + Resize(nearest_ceil, half_pixel) | Nearest_ceil index wrong after dilated ASPP + ConstantOfShape zero-add |
| ort_resize_linear_asymmetric | Resize(linear, asymmetric, 2x) + MatMul | ORT fuses linear asymmetric resize with downstream MatMul, reorders coordinate computation |

### Optimizer fusion bugs (18 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_ort_004 | Cast(float->int32->bool) chain | Optimizer fused float->int32->bool into float->bool, skipping truncation step |
| github_ort_005 | Manual LayerNorm (ReduceMean-Sub-Pow-ReduceMean-Add-Sqrt-Div-Mul-Add) | FusionLayerNormalization changes output for non-standard shapes under ORT_ENABLE_ALL |
| github_ort_006 | SkipLayerNorm with FP16 | FP16 fusion introduces up to 9.77 abs diff vs unfused path |
| github_ort_009 | ORT_ENABLE_EXTENDED optimizer Shape node sharing | Shared Shape output nodes for inputs with same symbolic dims but different runtime values |
| campaign_v2_0004 | Conv(3x3, pad=1) + BatchNorm(eps=1e-5) | ORT fuses Conv+BN by folding BN params into weights; fp32 rounding differs from sequential |
| campaign_v2_0005 | Sub(X,X) + TopK(k=1) | ORT const-folds Sub(X,X)->0; TopK tie-breaking on all-zero input differs from unfused path |
| campaign_v2_0006 | Gather(embedding) + Add + LayerNorm + TopK | ORT fuses Gather+Add+LayerNorm; near-tie values break differently for TopK |
| campaign_v2_0009 | Conv(3x3) + HardSwish + BatchNorm | ORT fuses Conv+HardSwish+BN into single kernel; operation reordering causes rounding diff |
| campaign_v2_0010 | Abs+Pow(2)+Sqrt chain after nearest_ceil Resize | ORT mishandles sqrt(x^2+eps) with abs; confused optimization |
| campaign_v2_0014 | CumSum(axis=1) + Gemm + ... + GELU + Sigmoid | ORT incorrectly simplifies CumSum output during graph rewriting |
| campaign_v2_0015 | Softsign + Mul + Greater + Where | ORT fuses Softsign->Mul->Where gating; branch selection logic diverges |
| campaign_v2_0016 | Resize + Unsqueeze/Squeeze(dims 4,5,6) + MatMul | ORT eliminates Unsqueeze/Squeeze identity chain, causes shape tracking errors |
| campaign_v2_0017 | Clip([-10,10]) + Exp + Log + Softmax | ORT cancels Clip->Exp->Log as identity, but Clip bounds make it UNSAFE (Clip output != X) |
| campaign_v2_0018 | Transpose-MatMul-Transpose double pattern + CumSum | ORT fuses double Transpose-MatMul-Transpose into merged MatMul; accumulated fp rounding |
| campaign_v2_0026 | MatMul + Mul(zeros) + Add(constant) | ORT eliminates mul-by-zero but ignores the subsequent Add that provides a needed constant offset |
| campaign_v2_0027 | Conv(shared weights) + ReLU + Conv(same weights) + Add(residual) | ORT incorrectly fuses residual double-Conv where both convolutions share the same weight tensor |
| campaign_v2_0044 | Resize + GlobalAvgPool + FC + Sigmoid (SE-block) + Resize | ORT fuses Sigmoid inside SE block + applies Resize fusion; order-of-ops sensitivity |
| campaign_v2_0049 | MatMul + Add + GELU + Gemm(gated) + Sigmoid + Mul(zeros) + Add | ORT fuses MatMul+Bias+GELU, then fails to correctly const-fold the zero-mask pattern |

### Convolution bugs (3 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_ort_011 | ConvTranspose(auto_pad=SAME_UPPER, stride=2) | auto_pad attribute ignored; used explicit pads=[0,0,0,0] instead |
| github_ort_012 | ConvTranspose(dilation=2, stride=2, output_padding=[1,1]) | ConvTranspose dilation>1 with output_padding gives wrong output values |
| campaign_v2_0047 | x^2->ReduceSum->Sqrt->Div (L2 norm) + Div+Mul+Add chain | ORT may fuse L2-norm into LpNormalization; subsequent chain diverges |

### Other operator bugs (8 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_ort_007 | TopK(equal values, axis=1, k=3) on GPU | GPU TopK non-deterministic on ties; CUDA bitonic sort race condition |
| github_ort_013 | InstanceNorm with FP16, large variance | GPU InstanceNorm accumulates in FP16; wrong for large-variance inputs |
| github_ort_014 | RoiAlign(avg, half_pixel, sampling_ratio=2) | half_pixel mode applied pixel offset incorrectly vs output_half_pixel |
| github_ort_015 | RoiAlign(max, half_pixel) | Max mode applied MAX before interpolation instead of after |
| github_ort_016 | GridSample(bicubic, border) at corner grid points | Bicubic+border clamped after neighbourhood lookup instead of per-sample |
| github_ort_017 | Einsum with uppercase labels like 'BIJ,BJK->BIK' | Einsum rejected uppercase labels; only lowercase was supported |
| github_ort_018 | ScatterND with duplicate indices on WebGPU | JSEP ScatterND race condition with duplicate indices on WebGPU |
| campaign_v2_0054 | MatMul(4D) x2 with identity-like weights | ORT merges back-to-back 4D MatMuls; shape/numerical divergence |

### Unresolved complex pattern bugs (15 bugs)

| Bug ID | Patterns | Root Cause |
|---|---|---|
| campaign_v2_0030 | Reciprocal+Mul+CumSum+LayerNorm+GELU | Rounding in reciprocal->mul->cumsum->layernorm->GELU chain |
| campaign_v2_0031 | Resize(nearest_ceil)+Transpose-cancel+MatMul(4D)+LogSumExp | Shape tracking through cancel patterns + LogSumExp diverges |
| campaign_v2_0032 | Mul(zeros)+Add+LayerNorm+MatMul+Dropout+Pow(1) | Constant folding + identity elimination interacts incorrectly with LayerNorm |
| campaign_v2_0035 | ReLU+Erf+Neg+Abs+MatMul(4D)+CumSum | Neg->Abs simplification in presence of identity folding |
| campaign_v2_0036 | log1p(|x|)*x+GELU+Max(consts)+LayerNorm x2+ReduceL2 | Log in log1p pattern + double LayerNorm diverges |
| campaign_v2_0037 | Resize(nearest_ceil)+MatMul x2+Where+Pad(edge) | Shape mishandling in back-to-back 4D MatMuls before Where masking |
| campaign_v2_0038 | Mul(1) x3+Sub(self)+TopK+Gather+LayerNorm | Identity elimination (mul-by-1, sub-self) interacts with TopK tie-breaking |
| campaign_v2_0039 | MatMul(4D)+ReduceSum(axis=2)+Concat+Resize(1x)+Concat | ReduceSum keepdims + broadcast Add + repeated Concat |
| campaign_v2_0043 | MatMul(1x1)+Resize(linear,half_pixel)+manual BN | JAX GPU JIT half_pixel linear resize precision (tested via ORT stand-in) |
| campaign_v2_0045 | Inception branches+Resize(cubic,half_pixel) | JAX GPU JIT cubic half_pixel diverges after stable softmax + CRMSNorm |
| ort_ada_layer_norm | X+Skip->LayerNorm->Mul(scale)->Add(shift) | ORT fuses adaptive LayerNorm (SkipLayerNorm) with incorrect parameter mapping |
| ort_reduce_sum_middle | ReduceSum(axis=1,keepdims=1)+Reshape+MatMul | ORT transposes axes incorrectly during reduce fusion |
| ort_relu_add_relu | ReLU->Add(negative Y)->ReLU | ORT fuses double-ReLU with residual, mishandling the intermediate negative Add |
| github_onnx_spec_001 | Cast(float->int32) with 0.5, 1.5, -0.5 | ONNX spec undefined for float-to-int rounding mode; ORT truncates, numpy rounds |
| github_onnx_spec_004 | TopK with NaN values, k=3 | Spec doesn't document NaN handling; ORT treats NaN > all finite (impl-defined) |

---

## torch.compile / Inductor (15 bugs)

### Fusion / codegen bugs (6 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_inductor_001 | BatchNorm2d(3), input (1,3,2,2) | Output shape [3,3,2,2] instead of [1,3,2,2] |
| github_inductor_002 | Conv2d+BatchNorm2d in train mode | Wrong batch statistics during training |
| github_inductor_003 | fuse_modules([Conv2d,BatchNorm2d,ReLU]) | Fusion silently replaced BatchNorm2d with Identity |
| github_inductor_005 | Conv2d+PixelShuffle(2) | PixelShuffle unregistered for meta tensors; wrong strides |
| github_inductor_008 | ConvTranspose2d with output_size=(0,0) | IndexError instead of proper ValueError for invalid output_size |
| torch_compile_avgpool_ceil | AvgPool2d(ceil_mode=True)+Conv2d | Inductor uses floor-mode padding for fused kernel when ceil_mode=True |

### Interpolation / spatial bugs (4 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_inductor_004 | F.interpolate(bilinear, align_corners=False) | AOT decomposition silently gives wrong bilinear values |
| github_inductor_006 | AvgPool(ceil_mode=1, count_include_pad=0) | Border windows divided by full kernel_size (9) instead of actual overlap (4) |
| github_inductor_007 | AvgPool(2x2, stride=2, count_include_pad=1) | GPU kernel returns undivided sum instead of sum/kernel_area |
| github_inductor_009 | GridSample with NaN grid + bilinear | NaN grid coordinates corrupt neighbouring valid outputs on GPU |

### Numerical precision bugs (3 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_inductor_010 | GridSample(bilinear, align_corners=True, identity grid) | FP rounding gives weights [1-eps,eps,0,0] instead of [1,0,0,0] |
| github_inductor_012 | Einsum('bij,bjk->bik') on MPS | MPS einsum batch matmul wrong on first call; uninitialized Metal command buffer |
| torch_compile_cast_roundtrip | x.to(int32).to(float32) chain | Inductor fuses int32 cast pair into no-op, losing floor-truncation |

### Out-of-bounds / safety bugs (2 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_inductor_011 | ScatterND with OOB indices | GPU kernel returns garbage for OOB instead of IndexError |
| torch_compile_matmul_4d | torch.matmul on [2,4,8,16]x[2,4,16,8] | Inductor generates incorrect tiling for 4D batch matmul |

---

## JAX/XLA (17 bugs)

### Resize operator (7 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| campaign_v2_0001 | jax.image.resize(bicubic, half_pixel_centers=True) | XLA JIT uses different cubic coefficients or pixel offset than eager |
| campaign_v2_0003 | Resize(cubic,half_pixel) + BatchNorm | XLA JIT fuses bicubic resize with BN scale/shift; operation reordering |
| campaign_v2_0008 | Resize(cubic,half_pixel) + Pad(edge) + Conv(dilated) | XLA JIT fuses dilated conv with edge-pad; wrong boundary computation |
| campaign_v2_0011 | Resize(linear,half_pixel, 2x) + L2 norm | XLA JIT bilinear interpolation precision differs from eager on GPU |
| campaign_v2_0013 | Resize(linear,asymmetric) + GroupNorm | XLA JIT fuses linear resize + group norm, rounding errors |
| campaign_v2_0021 | DepthToSpace(CRD,block=2) + Conv(1x1) + Resize(linear,align_corners) | XLA miscompiles DepthToSpace pixel-shuffle + align_corners resize |
| campaign_v2_0022 | Resize(linear,align_corners) + Resize(nearest,asymmetric) in sequence | XLA miscompiles dual Resize with different modes in same pipeline |

### Constant folding / optimization bugs (4 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| campaign_v2_0007 | Add(x, 0) before Softplus | XLA folds x+0->x, changes Softplus kernel path selection |
| campaign_v2_0025 | Where(all_True_constant, x, 0) + Conv + Softplus | XLA folds Where(true)->x; downstream ops handle folded constant differently |
| jax_xla_mul_zero_elim | x * 0 with large-valued x | XLA folds x*0 to zeros BEFORE evaluating x; misses NaN/Inf edge cases |
| jax_xla_self_sub_zero | (X - X) + Y | XLA folds X-X to scalar zero constant; shape or broadcast mismatch downstream |

### Numerical precision bugs (4 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| jax_xla_matmul_4d_batch | matmul on [2,4,8,16]x[2,4,16,8] under jax.jit | XLA GPU JIT miscompiles batched 4D matmul; output diverges from eager |
| jax_xla_cumsum_last_axis | cumsum(x, axis=3) under jax.jit on GPU | XLA GPU CumSum along last axis produces different values than eager |
| jax_xla_reduce_l2 | sqrt(sum(x^2, axis=-1)) under jax.jit | XLA uses fast RSqrt approximation for L2 norm; differs from eager |
| jax_xla_crms_norm | x / sqrt(mean(x^2) + eps) per channel | XLA fuses CRMS norm with subsequent ops; numerical divergence |

### Dual-path fusion bugs (2 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| campaign_v2_0023 | Dual Conv paths (one BN, one scale+bias) sharing weights + Add | XLA fuses shared-weight dual conv paths; intermediate results diverge |
| jax_xla_row_reduce_transpose | sum(axis=1) * x then transpose | XLA reorders transpose with reduce+multiply fusion; wrong layout |

---

## OpenVINO (14 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_openvino_001 | Softsign(x) = x/(1+|x|) | CPU JIT emitter returns all 1.0 regardless of input |
| github_openvino_003 | MatMul FP16 with N=2049 | GPU FP16 accumulation saturates at 2048.0 (max exact FP16 integer) |
| github_openvino_004 | Resize(cubic, half_pixel, dynamic shapes) | Cubic resize preprocessing wrong with dynamic-shape inputs; static OK |
| github_openvino_005 | Mul(NaN, 0) and Mul(inf, 0) | Optimizer folds x*0 to constant 0, violating IEEE 754 (NaN*0=NaN) |
| github_openvino_006 | Concat->Resize->Concat | Memory manager reuses Concat output buffer as Resize input; corrupts data |
| github_openvino_007 | TopK(K=128, N=5040) on NPU | NPU plugin throws ZE_RESULT_ERROR_UNKNOWN for large K values |
| github_openvino_008 | BatchNorm decomposition in Conv+BN fusion | Type mismatch on Multiply node during BN decomposition |
| github_openvino_009 | ScaledDotProductAttention (SDPA) | Scale applied to Q before matmul instead of to scores after |
| github_openvino_010 | DepthToSpace(mode=DCR/blocks_first) on NPU | NPU plugin crashes for DepthToSpace blocks_first mode |
| github_openvino_011 | AvgPool(ceil_mode=1, count_include_pad=0) int8 | Int8 kernel divides border windows by full kernel_size not actual overlap |
| github_openvino_012 | PRelu(slope=0.25) with negative inputs | ARM JIT uses output register instead of slope; computes x*x for negatives |
| github_openvino_013 | ConvTranspose(auto_pad=SAME_LOWER, stride=2) | SAME_LOWER used SAME_UPPER formula; 1-pixel shape error |
| github_openvino_014 | Einsum with scalar (rank-0) input | Frontend assumed rank >= 1; scalar input causes null-dereference crash |
| github_openvino_4 | MatMul [batch,2,1,16]x[batch,2,16,1], batch=4 | GPU plugin wrong for certain batch sizes with 4D batched matmul |

---

## TVM Relay (9 bugs)

| Bug ID | Op + Parameters | Root Cause |
|---|---|---|
| github_tvm_001 | BatchNorm eval mode | SimplifyInference skips BN when output stored before indexing; uses train stats |
| github_tvm_002 | FastMath tanh(NaN) | FastMath reversed clamp args; min(NaN,9)=9 at C level, so tanh(NaN)=1.0 |
| github_tvm_003 | Resize(nearest, align_corners, round_prefer_ceil) | TOPI validation blocks nearest+align_corners+round_prefer_ceil combo |
| github_tvm_004 | Resize(cubic, half_pixel, cubic_coeff_a=-0.5) | ONNX importer cubic interpolation diverges from ORT and PyTorch bicubic |
| github_tvm_005 | Dilated conv + regular conv branches, batch=2 | NCHWc layout transform at opt_level=3 gives wrong per-sample outputs |
| github_tvm_006 | PRelu per-channel slopes broadcast | Hardcoded channel axis=1 (NCHW); NHWC slope broadcast applied to H not C |
| github_tvm_007 | ConvTranspose(stride=2, output_padding=[1,1]) | Ignores output_padding; 9x9 instead of 10x10 |
| github_tvm_008 | InstanceNorm with small spatial dims (2x2) | GPU parallel reduction wrong mean/variance for small spatial dims |
| github_tvm_009 | ScatterND with diagonal indices on CUDA | CUDA kernel missing return statement; thread corrupts memory |

---

## ONNX Spec ambiguities (8 bugs)

| Bug ID | Issue | Root Cause |
|---|---|---|
| github_onnx_spec_001 | Cast(float->int32) rounding | Spec undefined for float-to-int rounding; ORT truncates, numpy banker's-rounds |
| github_onnx_spec_002 | Resize output_size for fractional dims | floor vs round undefined for 7*1.5=10.5; ORT=10 |
| github_onnx_spec_003 | Resize pytorch_half_pixel when output_size=1 | Formula gives x_orig=-0.5; clip to 0 = first pixel instead of center |
| github_onnx_spec_004 | TopK NaN handling | NaN comparison undefined; ORT treats NaN > all finite values |
| github_onnx_spec_005 | TopK equal-value tie-breaking | Initially ambiguous; now spec says lower index wins on ties |
| github_onnx_spec_006 | CumSum axis parameter | Axis marked optional, causes undefined behavior when absent |
| github_onnx_spec_007 | Resize nearest half_pixel rounding | ORT returns wrong index; spec ambiguous on half_pixel rounding |
| github_onnx_spec_13 | Resize opset-12->13 tf_half_pixel_for_nn | CompatibleAdapter mishandles deprecated attribute |
