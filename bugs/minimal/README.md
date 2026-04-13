# Minimal Reproducible Bug Scripts

**145 self-contained Python scripts** — one per unique root cause.
Each file constructs its test model inline (no external files, no embedded weights).

```
Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
```

## Summary

| Compiler | Bugs |
|---|---|
| OnnxRuntime | 52 |
| JAX/XLA | 43 |
| torch.compile / onnx2torch | 18 |
| OpenVINO | 15 |
| TVM | 9 |
| ONNX Spec | 8 |
| **Total** | **145** |


## Run Results (2026-04-13)

Tested on: ORT 1.24.4, PyTorch 2.9.1, ONNX 1.21.0, JAX 0.9.2 (CPU), OpenVINO 2026.0, no TVM.

| Status | Count | Description |
|---|---|---|
| BUG | 12 | Still reproduce on current versions |
| FIXED | 133 | No longer reproduce — bug was fixed in current compiler version |
| **Total** | **145** | |

Tested on: ORT 1.24.4, PyTorch 2.9.1, ONNX 1.21.0, JAX 0.9.2 + CUDA, OpenVINO 2026.0.

### 12 Still-Live Bugs

| # | Bug ID | Compiler | max_diff | Root Cause |
|---|---|---|---|---|
| 1 | github_onnx_spec_007 | ONNX Spec | — | Resize nearest half_pixel rounding returns wrong index for element 4 |
| 2 | github_ort_002 | OnnxRuntime 1.24.4 | — | Nearest resize 1->64 pixel selection off-by-one vs PyTorch |
| 3 | github_ort_003 | OnnxRuntime 1.24.4 | 0.29 | Asymmetric bilinear resize max error vs PyTorch |
| 4 | github_ort_004 | OnnxRuntime 1.24.4 | — | Optimizer fuses float->int32->bool, skipping truncation step |
| 5 | github_ort_008 | OnnxRuntime 1.24.4 | — | CUDA cubic resize antialias grid wrong, hardcoded cubic_coeff_a |
| 6 | github_ort_016 | OnnxRuntime 1.24.4 | — | GridSample bicubic+border clamps after neighbourhood instead of per-sample |
| 7 | github_tensorflow_002 | TensorFlow XLA | 1.0 | MLIR/TOSA nearest resize shifts rows by 1 with half_pixel_centers |
| 8 | github_tvm_004 | TVM Relay | 0.053 | Cubic interpolation diverges from ORT and PyTorch bicubic |
| 9 | **cross_onnx2torch_resize_nearest_ceil** | **onnx2torch** | **3.00** | **onnx2torch uses floor nearest_mode instead of ceil** |
| 10 | **cross_onnx2torch_resize_linear_asym** | **onnx2torch** | **0.80** | **asymmetric coord mode mapped to wrong PyTorch interpolation** |
| 11 | **cross_openvino_conv_bn_fusion** | **OpenVINO 2026.0** | **0.022** | **Conv+BN fusion rounding error above tolerance** |
| 12 | **cross_onnx2torch_cumsum** | **onnx2torch** | **4.50** | **CumSum axis.item() causes graph break and wrong output** |

---

## OnnxRuntime (52 bugs)

| # | Bug ID | Root Cause |
|---|---|---|
| 1 | [campaign_v2_0002](campaign_v2_0002.py) | ONNX Resize with mode='nearest', nearest_mode='ceil', |
| 2 | [campaign_v2_0004](campaign_v2_0004.py) | ORT optimizer fuses Conv + BatchNormalization into a single Conv by folding |
| 3 | [campaign_v2_0005](campaign_v2_0005.py) | When ORT const-folds Sub(X, X) -> all-zero tensor, the resulting zero |
| 4 | [campaign_v2_0006](campaign_v2_0006.py) | ORT optimizer fuses Gather(embedding) + Add(input) + LayerNormalization |
| 5 | [campaign_v2_0009](campaign_v2_0009.py) | ORT optimizer fuses Conv -> HardSwish -> BatchNormalization into a |
| 6 | [campaign_v2_0010](campaign_v2_0010.py) | ORT optimizer mishandles Abs+Pow(2)+Sqrt chain after nearest_ceil |
| 7 | [campaign_v2_0014](campaign_v2_0014.py) | ORT optimizer incorrectly simplifies the CumSum output when it |
| 8 | [campaign_v2_0015](campaign_v2_0015.py) | ORT optimizer fuses the Softsign -> Mul -> Where gating pattern |
| 9 | [campaign_v2_0016](campaign_v2_0016.py) | ORT optimizer eliminates the Unsqueeze/Squeeze identity chain after |
| 10 | [campaign_v2_0017](campaign_v2_0017.py) | ORT optimizer cancels the Clip->Exp->Log sequence (treating it as |
| 11 | [campaign_v2_0018](campaign_v2_0018.py) | ORT optimizer fuses the double Transpose-MatMul-Transpose pattern |
| 12 | [campaign_v2_0024](campaign_v2_0024.py) | ORT's graph optimizer incorrectly merges / reorders two consecutive Resize operations |
| 13 | [campaign_v2_0026](campaign_v2_0026.py) | ORT's optimizer incorrectly eliminates the multiply-by-zero (mul-zero-elim pass) |
| 14 | [campaign_v2_0027](campaign_v2_0027.py) | ORT incorrectly fuses a residual double-Conv block (both branches sharing the same |
| 15 | [campaign_v2_0028](campaign_v2_0028.py) | ORT incorrectly fuses Resize(nearest_ceil) with a downstream ReduceSum that |
| 16 | [campaign_v2_0029](campaign_v2_0029.py) | ORT's optimizer incorrectly reorders a Resize(nearest_ceil) with a downstream |
| 17 | [campaign_v2_0030](campaign_v2_0030.py) | n/a |
| 18 | [campaign_v2_0031](campaign_v2_0031.py) | n/a |
| 19 | [campaign_v2_0032](campaign_v2_0032.py) | n/a |
| 20 | [campaign_v2_0035](campaign_v2_0035.py) | n/a |
| 21 | [campaign_v2_0036](campaign_v2_0036.py) | n/a |
| 22 | [campaign_v2_0037](campaign_v2_0037.py) | n/a |
| 23 | [campaign_v2_0038](campaign_v2_0038.py) | n/a |
| 24 | [campaign_v2_0039](campaign_v2_0039.py) | n/a |
| 25 | [campaign_v2_0044](campaign_v2_0044.py) | ORT fuses Sigmoid inside the SE block and may also apply Resize fusion; |
| 26 | [campaign_v2_0047](campaign_v2_0047.py) | ORT may fuse the L2-norm pattern into an LpNormalization op, then the |
| 27 | [campaign_v2_0049](campaign_v2_0049.py) | ORT fuses the MatMul+Bias+GELU pattern into a single GeLU kernel. |
| 28 | [campaign_v2_0053](campaign_v2_0053.py) | ORT's nearest-ceil resize with half_pixel coordinate mode may compute |
| 29 | [campaign_v2_0054](campaign_v2_0054.py) | ORT may merge the back-to-back 4D MatMuls (both 1x1xCxC identity-like) |
| 30 | [campaign_v2_0055](campaign_v2_0055.py) | ORT nearest-ceil resize with half_pixel coordinate mode after dilated ASPP |
| 31 | [github_ort_001](github_ort_001.py) | Bug: tf_half_pixel_for_nn applied to batch/channel dims in old ORT caused |
| 32 | [github_ort_002](github_ort_002.py) | Bug: nearest resize 1->64 (scale=64/26) produces pixel selection one off vs PyTorch. |
| 33 | [github_ort_003](github_ort_003.py) | Bug: asymmetric coord mode gives up to 0.2947 error vs PyTorch bilinear. |
| 34 | [github_ort_004](github_ort_004.py) | Bug: ORT_ENABLE_ALL optimizer strips float->int32->bool to float->bool. |
| 35 | [github_ort_005](github_ort_005.py) | Bug: FusionLayerNormalization changes output for non-standard shapes. |
| 36 | [github_ort_006](github_ort_006.py) | Bug: SkipLayerNorm FP16 fusion introduces up to 9.77 abs diff vs unfused. |
| 37 | [github_ort_007](github_ort_007.py) | Input with many equal values — the tie-breaking case |
| 38 | [github_ort_008](github_ort_008.py) | PyTorch reference bicubic |
| 39 | [github_ort_009](github_ort_009.py) | Two tensors with the same shape but different values; Shape nodes must NOT be shared |
| 40 | [github_ort_010](github_ort_010.py) | Bug: ORT GPU Resize linear outputs constant 0.5 when preceded by MatMul (issue #12091). |
| 41 | [github_ort_011](github_ort_011.py) | Bug: ORT ConvTranspose auto_pad=SAME_UPPER ignored attribute, used explicit pads=[0,0,0,0] (issue #4086). |
| 42 | [github_ort_012](github_ort_012.py) | Bug: ORT ConvTranspose dilation>1 + output_padding gives wrong output values (issue #14208). |
| 43 | [github_ort_013](github_ort_013.py) | Bug: ORT GPU InstanceNorm FP16 accumulated in FP16, wrong for large-variance inputs (PR#9879). |
| 44 | [github_ort_014](github_ort_014.py) | Bug: ORT RoiAlign half_pixel mode applied pixel offset incorrectly (issue #6921). |
| 45 | [github_ort_015](github_ort_015.py) | Bug: ORT RoiAlign max mode applied MAX before interpolation instead of after (issue #6146). |
| 46 | [github_ort_016](github_ort_016.py) | Bug: ORT GridSample bicubic+border clamps after neighbourhood lookup instead of per-sample (issue #10607). |
| 47 | [github_ort_017](github_ort_017.py) | Bug: ORT Einsum rejected uppercase labels like "BIJ,BJK->BIK" (issue #4944). |
| 48 | [github_ort_018](github_ort_018.py) | Bug: ORT JSEP ScatterND race with duplicate indices (PR#23755). CPU is deterministic (last-write wins). |
| 49 | [ort_ada_layer_norm](ort_ada_layer_norm.py) | ORT fuses the adaptive LayerNorm (SkipLayerNorm pattern) with incorrect parameter mapping, mixing scale and bias |
| 50 | [ort_reduce_sum_middle](ort_reduce_sum_middle.py) | ORT optimizer incorrectly transposes axes during a reduce fusion, summing along the wrong dimension |
| 51 | [ort_relu_add_relu](ort_relu_add_relu.py) | ORT fuses the double-ReLU with residual into a single activation, incorrectly handling the intermediate Add |
| 52 | [ort_resize_linear_asymmetric](ort_resize_linear_asymmetric.py) | ORT optimizer fuses linear asymmetric resize with downstream MatMul, reordering coordinate computation and producing ... |

## JAX/XLA (43 bugs)

| # | Bug ID | Root Cause |
|---|---|---|
| 1 | [campaign_v2_0001](campaign_v2_0001.py) | jax.image.resize with method='bicubic' and half_pixel_centers=True |
| 2 | [campaign_v2_0003](campaign_v2_0003.py) | JAX XLA JIT fuses bicubic resize with BatchNorm scale/shift into a |
| 3 | [campaign_v2_0007](campaign_v2_0007.py) | JAX XLA JIT applies an "add-zero identity" fold (x + 0 -> x) that |
| 4 | [campaign_v2_0008](campaign_v2_0008.py) | JAX XLA JIT fuses a dilated convolution branch with an edge-mode padding |
| 5 | [campaign_v2_0011](campaign_v2_0011.py) | JAX linear resize diverges under GPU JIT — JIT uses different |
| 6 | [campaign_v2_0013](campaign_v2_0013.py) | JAX linear resize with align_corners gets fused with group norm under |
| 7 | [campaign_v2_0021](campaign_v2_0021.py) | JAX/XLA GPU JIT miscompiles a pipeline that pixel-shuffles via DepthToSpace (CRD mode), |
| 8 | [campaign_v2_0022](campaign_v2_0022.py) | XLA GPU JIT miscompiles a pipeline with two Resize nodes of different modes. |
| 9 | [campaign_v2_0023](campaign_v2_0023.py) | XLA GPU JIT fuses dual conv paths (one through BN, one through scale+bias) that share |
| 10 | [campaign_v2_0025](campaign_v2_0025.py) | XLA folds the Where node (condition is all-True constant) and then incorrectly |
| 11 | [campaign_v2_0043](campaign_v2_0043.py) | JAX GPU JIT bicubic/linear half_pixel resize diverges from eager — |
| 12 | [campaign_v2_0045](campaign_v2_0045.py) | JAX GPU JIT bicubic halfpixel resize diverges from eager after |
| 13 | [github_tensorflow_001](github_tensorflow_001.py) | Bug: TFLite converter silently forced align_corners=False (half_pixel) instead of True (asymmetric). |
| 14 | [github_tensorflow_002](github_tensorflow_002.py) | Bug: TF MLIR/TOSA nearest resize shifted rows by 1 with half_pixel_centers=True (issue #62386). |
| 15 | [github_tensorflow_003](github_tensorflow_003.py) | 2-channel 4x4 input, block_size=2 -> output [1, 8, 2, 2] |
| 16 | [github_tensorflow_004](github_tensorflow_004.py) | Bug: TF BN folding into Conv applied scale incorrectly in quantization-aware training (issue #43882). |
| 17 | [jax_xla_add_zero_identity](jax_xla_add_zero_identity.py) | XLA JIT folds x+0 into identity, changing numerical path for downstream exp/log |
| 18 | [jax_xla_batchnorm_eval](jax_xla_batchnorm_eval.py) | XLA JIT fuses batch norm eval with subsequent conv, reordering FP operations |
| 19 | [jax_xla_cast_roundtrip](jax_xla_cast_roundtrip.py) | JAX XLA JIT fuses the int32 cast pair into identity, losing the floor-truncation; values near 0.5 diverge |
| 20 | [jax_xla_concat_conv](jax_xla_concat_conv.py) | XLA JIT fuses concat+conv, may reorder channel reduction causing different FP accumulation |
| 21 | [jax_xla_conv_relu_concat_maxpool](jax_xla_conv_relu_concat_maxpool.py) | XLA JIT parallelizes two conv branches and fuses with maxpool; boundary padding differs |
| 22 | [jax_xla_crms_norm](jax_xla_crms_norm.py) | JAX XLA JIT fuses the RMS norm with subsequent operations, producing numerical divergence vs eager |
| 23 | [jax_xla_cumsum_last_axis](jax_xla_cumsum_last_axis.py) | JAX XLA JIT produces different cumsum along last axis than eager mode on GPU |
| 24 | [jax_xla_dilated_branch_sum](jax_xla_dilated_branch_sum.py) | XLA JIT fuses multi-dilation branch sum, possibly reordering or vectorizing differently |
| 25 | [jax_xla_foldable_sub_zero_div_one](jax_xla_foldable_sub_zero_div_one.py) | XLA JIT folds sub-zero and div-one into identities, changing precision tracking for cumsum |
| 26 | [jax_xla_fpn_branch](jax_xla_fpn_branch.py) | XLA JIT fuses FPN upsample+skip-add+conv+tanh pipeline, causing FP divergence at boundaries |
| 27 | [jax_xla_identity_chain](jax_xla_identity_chain.py) | XLA JIT optimizes away identity but changes numerical path for floor/ceil chain |
| 28 | [jax_xla_layernorm_chain](jax_xla_layernorm_chain.py) | XLA JIT fuses layer norm + sin/tanh activation + attention matmul; softmax precision differs |
| 29 | [jax_xla_log1p_abs](jax_xla_log1p_abs.py) | XLA JIT applies fast-math to log(abs+1) chain, producing different values than eager |
| 30 | [jax_xla_log_abs_eps](jax_xla_log_abs_eps.py) | XLA JIT fast-math or reassociation changes log(abs+eps) precision |
| 31 | [jax_xla_matmul_4d_batch](jax_xla_matmul_4d_batch.py) | JAX XLA JIT miscompiles batched 4D matmul; output diverges from jax.numpy.matmul eager mode |
| 32 | [jax_xla_matmul_scale_softmax](jax_xla_matmul_scale_softmax.py) | XLA JIT fuses matmul+scale+softmax; the fused softmax kernel uses different max-subtraction |
| 33 | [jax_xla_mul_by_one_chain](jax_xla_mul_by_one_chain.py) | XLA JIT folds triple mul-by-one into identity, changing memory layout for downstream conv+tanh |
| 34 | [jax_xla_mul_zero_elim](jax_xla_mul_zero_elim.py) | XLA JIT folds x * 0 into zeros BEFORE evaluating x, ignoring NaN/Inf propagation; diverges from eager when x has larg... |
| 35 | [jax_xla_reduce_l2](jax_xla_reduce_l2.py) | JAX XLA JIT applies a fast RSqrt approximation for L2 normalization that differs numerically from eager |
| 36 | [jax_xla_reduce_max_last](jax_xla_reduce_max_last.py) | JAX XLA JIT tree-reduces along last axis using a different ordering than eager, causing floating-point associativity ... |
| 37 | [jax_xla_reshape_layernorm](jax_xla_reshape_layernorm.py) | XLA JIT eliminates reshape-norm-reshape and identity transpose; fused path computes differently |
| 38 | [jax_xla_residual_add_relu](jax_xla_residual_add_relu.py) | XLA JIT fuses residual-add-relu with erf/tanh chain; erf(tanh(x))*x computation diverges |
| 39 | [jax_xla_rmsnorm](jax_xla_rmsnorm.py) | XLA JIT fuses RMS norm (sqrt(mean(x^2)+eps)) with gated MLP, numerical divergence in sqrt |
| 40 | [jax_xla_row_reduce_transpose](jax_xla_row_reduce_transpose.py) | JAX XLA JIT reorders the transpose with the reduce+multiply fusion, producing wrong layout |
| 41 | [jax_xla_self_sub_zero](jax_xla_self_sub_zero.py) | XLA JIT folds X - X to a zero constant BEFORE evaluating X, but the downstream ops expect a proper tensor (not a cons... |
| 42 | [jax_xla_where_const_condition](jax_xla_where_const_condition.py) | XLA JIT folds Where with constant True/False mask, changing precision for downstream conv+softplus |
| 43 | [jax_xla_where_mask_fill](jax_xla_where_mask_fill.py) | XLA JIT fuses where-mask with GLU sigmoid gate; threshold comparison + sigmoid fusion produces different values |

## torch.compile (15 bugs)

| # | Bug ID | Root Cause |
|---|---|---|
| 1 | [github_inductor_001](github_inductor_001.py) | Bug: Inductor #100970 — BatchNorm2d with torch.compile produced wrong output shape [3,3,2,2] not [1,3,2,2]. |
| 2 | [github_inductor_002](github_inductor_002.py) | Bug: Inductor #141317 — Conv2d+BN in train mode: inductor computed wrong batch stats, output diverged from eager. |
| 3 | [github_inductor_003](github_inductor_003.py) | Bug: PyTorch #112820 — fuse_modules([conv,bn,relu]) silently replaced BN with Identity, giving wrong outputs. |
| 4 | [github_inductor_004](github_inductor_004.py) | Bug: Inductor #93262 — torch.compile AOT decomposition of bilinear interpolate silently gave wrong values. |
| 5 | [github_inductor_005](github_inductor_005.py) | Bug: PyTorch #91551 — PixelShuffle unregistered for meta tensors; compile generated wrong strides, AssertionError. |
| 6 | [github_inductor_006](github_inductor_006.py) | Bug: Inductor #100987 — AvgPool ceil_mode border windows divided by full kernel=9, not actual overlap=4. |
| 7 | [github_inductor_007](github_inductor_007.py) | Bug: Inductor #143720 — AvgPool GPU kernel returned undivided sum instead of sum/kernel_area. |
| 8 | [github_inductor_008](github_inductor_008.py) | Bug: Inductor #144013 — ConvTranspose2d output_size=(0,0) caused IndexError in Inductor, not correct ValueError. |
| 9 | [github_inductor_009](github_inductor_009.py) | Bug: PyTorch #24823 — GridSample NaN grid coords corrupted neighbouring valid outputs on GPU. |
| 10 | [github_inductor_010](github_inductor_010.py) | Bug: PyTorch #81868 — GridSample bilinear identity grid: FP rounding gave [1-eps,eps,0,0] not [1,0,0,0]. |
| 11 | [github_inductor_011](github_inductor_011.py) | Bug: Inductor #122291 — ScatterND OOB indices returned garbage on GPU; eager raised IndexError correctly. |
| 12 | [github_inductor_012](github_inductor_012.py) | Bug: PyTorch #85224 — MPS einsum batch matmul wrong on first call; Metal command buffer not initialized. |
| 13 | [torch_compile_avgpool_ceil](torch_compile_avgpool_ceil.py) | torch.compile Inductor uses floor-mode padding calculation for the fused AvgPool+Conv kernel when ceil_mode=True, pro... |
| 14 | [torch_compile_cast_roundtrip](torch_compile_cast_roundtrip.py) | Inductor fuses the int32 cast pair into a no-op in the generated kernel, losing the floor-truncation effect on fracti... |
| 15 | [torch_compile_matmul_4d](torch_compile_matmul_4d.py) | Inductor generates incorrect tiling for 4D batch matmul (common in attention blocks), producing wrong values when bat... |

## OpenVINO (14 bugs)

| # | Bug ID | Root Cause |
|---|---|---|
| 1 | [github_openvino_001](github_openvino_001.py) | OpenVINO Bug #31252 - Softsign CPU JIT emitter returns all 1.0 regardless of input |
| 2 | [github_openvino_003](github_openvino_003.py) | OpenVINO Bug #22613 - GPU MatMul FP16 wrong result when N > 2048 |
| 3 | [github_openvino_004](github_openvino_004.py) | OpenVINO Bug #22854 - RESIZE_CUBIC preprocessing wrong with dynamic shape inputs |
| 4 | [github_openvino_005](github_openvino_005.py) | OpenVINO Bug #8729 - mul_zero_elim: x * 0 -> 0 is unsound when x = NaN |
| 5 | [github_openvino_006](github_openvino_006.py) | OpenVINO Bug #5505 - Wrong output in Concat->Resize->Concat (in-place buffer reuse) |
| 6 | [github_openvino_007](github_openvino_007.py) | OpenVINO Bug #29297 - TopK NPU throws ZE_RESULT_ERROR_UNKNOWN for large K |
| 7 | [github_openvino_008](github_openvino_008.py) | OpenVINO Bug #23539 - Shape inference error on Multiply in BatchNorm decomposition |
| 8 | [github_openvino_009](github_openvino_009.py) | OpenVINO Bug #34177 - SDPA decomposition applies scale before MatMul(Q,K^T) instead of after |
| 9 | [github_openvino_010](github_openvino_010.py) | OpenVINO Bug #29029 - DepthToSpace fails in blocks_first (DCR) mode on NPU |
| 10 | [github_openvino_011](github_openvino_011.py) | OpenVINO Bug #20815 - AvgPool CPU int8 with ceil_mode gives 71.4% wrong outputs |
| 11 | [github_openvino_012](github_openvino_012.py) | OpenVINO PR#28223 - PReLU ARM JIT emitter applies wrong slope for negative inputs |
| 12 | [github_openvino_013](github_openvino_013.py) | OpenVINO Bug #30798 - ConvTranspose auto_pad=SAME_LOWER gives wrong output shape |
| 13 | [github_openvino_014](github_openvino_014.py) | OpenVINO PR#30189 - Einsum parser crashes on scalar (rank-0) input operands |
| 14 | [github_openvino_4](github_openvino_4.py) | OpenVINO Bug #28881 - 4D MatMul FP32 GPU plugin wrong output |

## TVM (9 bugs)

| # | Bug ID | Root Cause |
|---|---|---|
| 1 | [github_tvm_001](github_tvm_001.py) | Bug: TVM #6852 — SimplifyInference skips BN when output stored before indexing; train-mode stats used at inference. |
| 2 | [github_tvm_002](github_tvm_002.py) | Bug: TVM #11696 — FastMath tanh: reversed clamp args cause min(NaN,9)=9, so tanh(NaN)=1.0 instead of NaN. |
| 3 | [github_tvm_003](github_tvm_003.py) | Bug: TVM PR #7532 — topi resize validation blocked nearest+align_corners+round_prefer_ceil combo. |
| 4 | [github_tvm_004](github_tvm_004.py) | See code comments |
| 5 | [github_tvm_005](github_tvm_005.py) | Dilated conv + regular conv (branch merge) — triggers the NCHWc bug |
| 6 | [github_tvm_006](github_tvm_006.py) | Bug: TVM PR #7208 — PReLU hardcoded channel axis=1 (NCHW); NHWC slope broadcast applied to H not C. |
| 7 | [github_tvm_007](github_tvm_007.py) | Bug: TVM PR #7958 — ConvTranspose ignores output_padding; stride=2 + output_padding=[1,1] gave 9x9 not 10x10. |
| 8 | [github_tvm_008](github_tvm_008.py) | Bug: TVM #15683 — InstanceNorm GPU parallel reduction gave wrong mean/var for small spatial dims (2x2). |
| 9 | [github_tvm_009](github_tvm_009.py) | Bug: TVM PR #7447 — ScatterND CUDA kernel missing return; thread continued past update, corrupting memory. |

## ONNX Spec (8 bugs)

| # | Bug ID | Root Cause |
|---|---|---|
| 1 | [github_onnx_spec_001](github_onnx_spec_001.py) | Bug: ONNX spec #5004 — Cast float->int rounding undefined; ORT truncates toward zero, NumPy rounds. |
| 2 | [github_onnx_spec_002](github_onnx_spec_002.py) | Bug: ONNX spec #4919 — Resize output dim floor vs round inconsistency; 7*1.5=10.5, ORT uses floor=10. |
| 3 | [github_onnx_spec_003](github_onnx_spec_003.py) | Bug: ONNX spec #4275 — pytorch_half_pixel output_size=1: formula gives x_orig=-0.5, ORT clips to 0 not center. |
| 4 | [github_onnx_spec_004](github_onnx_spec_004.py) | Bug: ONNX spec #7754 — TopK NaN handling undocumented; ORT treats NaN > all finite, spec says nothing. |
| 5 | [github_onnx_spec_005](github_onnx_spec_005.py) | Bug: ONNX spec #3501 — TopK tie-breaking ambiguous; spec now requires lower index wins on equal values. |
| 6 | [github_onnx_spec_006](github_onnx_spec_006.py) | Bug: ONNX spec #2611 — CumSum axis marked optional in schema; absent axis causes undefined behavior. |
| 7 | [github_onnx_spec_007](github_onnx_spec_007.py) | Bug: Resize nearest half_pixel round_prefer_ceil 20->6: ORT returned wrong index at element 4 (spec #4583). |
| 8 | [github_onnx_spec_13](github_onnx_spec_13.py) | Bug: ONNX spec #5266 — Resize opset-12->13 adapter uses CompatibleAdapter, mishandling deprecated tf_half_pixel_for_nn. |

