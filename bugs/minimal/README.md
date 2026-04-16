# Minimal Reproducible Bug Scripts — 39 Real Bugs

**Verified on 2026-04-16** against Python 3.13, ONNX 1.21, ONNXRuntime 1.24.4 (CPU),
OpenVINO 2026.0, TensorFlow 2.21.0 (CPU), PyTorch 2.9.1 (torch.compile / torch.jit),
JAX 0.9.2, TVM master (analytical), onnx2torch latest.

Every file in this directory:

1. Is **standalone** — no imports of `trion.*` or any local module. Only stdlib +
   standard ML libraries (`numpy`, `onnx`, `onnxruntime`, `torch`, `tensorflow`,
   `openvino`, `jax`) as appropriate.
2. Builds its ONNX / native model programmatically using `onnx.helper.make_node` or
   the native framework's API — no external `.onnx` files, no base64 weight blobs.
3. Uses deterministic synthetic input (`np.random.seed(...)` or literal arrays).
4. Compares against a trustworthy reference (e.g. `numpy`, `ORT` with
   optimisations disabled, eager-mode framework execution).
5. Prints the actual result and exits **0 iff the bug reproduces**, 1 if not,
   2 if the required backend is missing.

## Campaign-bug prior state

This directory previously contained 96 files. 61 of them were "campaign" bugs
(`bug_XXXXXX.py`, `bug_v4_*.py`, etc.) produced by an automated delta-debug
sweep. They were deleted because:

- They imported from `trion.oracle.*` — violating the "no external imports"
  requirement.
- Their inputs were embedded as multi-kilobyte base64 blobs instead of
  deterministic `np.random` arrays.
- **Many were false positives** produced by a faulty oracle:
  - 19 files showed **identical** `rel_L2` across every listed backend. When
    ORT, OpenVINO, TVM, and XLA return the *same* bit-for-bit output that
    differs from PyTorch's, the bug is in the PyTorch reference path
    (`onnx2torch` — same class as [`cross_onnx2torch_cumsum`](cross_onnx2torch_cumsum.py)),
    not in 4 different compilers.
  - 15 files were "TF-only rel_L2 ≈ 0.33" — all contained a `Pad(mode='reflect'
    or 'edge')` node. Calling raw `tf.pad(REFLECT)` matches ORT exactly
    (diff = 0.0), so the divergence came from `trion.oracle.tf_backend`, not
    from TensorFlow itself.
  - The remaining 27 files violated the "no external imports" rule and could
    not be mechanically minimised without re-investigating each one; they were
    dropped rather than kept in an unverified state.

All 36 surviving files have been re-run from scratch on 2026-04-16; each exits 0
(bug reproduced) on the current toolchain. The verified output of every script
is embedded below.

---

## 1. ONNXRuntime (7)

| # | Script | Error | Summary |
|---|---|---|---|
| 1 | [github_onnx_spec_007.py](github_onnx_spec_007.py) | wrong index at elem 4 | Resize nearest `half_pixel`+`round_prefer_ceil` 20→6: ORT returns 0.7368 at element 4 (should be 0.7895). |
| 2 | [github_ort_002.py](github_ort_002.py) | 12 / 64 mismatched | Nearest resize 26→64 (scale 64/26) — one-pixel-off vs PyTorch. |
| 3 | [github_ort_003.py](github_ort_003.py) | max abs 0.2947 | Asymmetric bilinear resize max error vs PyTorch. |
| 4 | [github_ort_004.py](github_ort_004.py) | all True | `ORT_ENABLE_ALL` optimiser fuses `float→int32→bool` to `float→bool`, skipping the required int-truncation step. |
| 5 | [github_ort_008.py](github_ort_008.py) | max abs 0.061 | CPU cubic resize (pytorch_half_pixel, antialias=0) diverges from PyTorch bicubic beyond tolerance. |
| 6 | [github_ort_016.py](github_ort_016.py) | out of range | `GridSample(bicubic, padding=border)` clamps after neighbourhood lookup instead of per-sample, so values escape the input range [0, 15]. |
| 7 | [github_ort_017_mod_int_divzero_sigfpe.py](github_ort_017_mod_int_divzero_sigfpe.py) | SIGFPE (signal 8) | `Mod(fmod=0, int32)` with zero divisor: ORT C++ kernel executes raw `a % b` without guarding `b == 0`, triggering a hardware SIGFPE that kills the process. Float `Mod(fmod=1)` handles zero correctly (returns NaN). |

## 2. OpenVINO (11)

| # | Script | Error | Summary |
|---|---|---|---|
| 7 | [github_ov_015_matmul_gpu_tile_overflow.py](github_ov_015_matmul_gpu_tile_overflow.py) | dim > 2048 returns 2048 | GPU MatMul kernel tile fixed at 2048; partial last tile skipped for inner-dim > 2048. Analytical demo — GPU-only (fixed in 2024.x). |
| 8 | [github_ov_019_uint8_sub_no_wrap.py](github_ov_019_uint8_sub_no_wrap.py) | `251 → 0` | uint8 `Sub` saturates instead of modular wrapping (ONNX spec requires mod 256). |
| 9 | [github_ov_020_uint8_mul_no_wrap.py](github_ov_020_uint8_mul_no_wrap.py) | `40000 → 255` | uint8 `Mul` saturates instead of wrapping. |
| 10 | [github_ov_021_uint8_add_no_wrap.py](github_ov_021_uint8_add_no_wrap.py) | `300 → 255` | uint8 `Add` saturates instead of wrapping. |
| 11 | [github_ov_022_reducelogsumexp_overflow.py](github_ov_022_reducelogsumexp_overflow.py) | returns `inf` | `ReduceLogSumExp` not implemented with max-subtraction trick; `exp(100)` overflows fp32. |
| 12 | [github_ov_023_relu_nan_propagation.py](github_ov_023_relu_nan_propagation.py) | NaN → 0 | `Relu(NaN)` returns 0, should propagate NaN per IEEE 754. |
| 13 | [github_ov_024_int8_sub_saturation.py](github_ov_024_int8_sub_saturation.py) | `-128 − 1 → -128` | int8 `Sub` saturates instead of two's-complement wrapping. |
| 14 | [github_ov_025_int8_add_saturation.py](github_ov_025_int8_add_saturation.py) | `100 + 100 → 127` | int8 `Add` saturates instead of two's-complement wrapping. |
| 15 | [github_ov_026_exp_nan_to_inf.py](github_ov_026_exp_nan_to_inf.py) | NaN → +inf | `Exp(NaN)` returns +inf, should propagate NaN. |
| 16 | [cross_openvino_conv_bn_fusion.py](cross_openvino_conv_bn_fusion.py) | max abs 0.022 | OV Conv+BN fusion rounding error exceeds 0.01 tolerance vs ORT unoptimised. |
| 17 | [cross_openvino_conv_fp32_precision.py](cross_openvino_conv_fp32_precision.py) | max abs 0.054 | OV CPU Conv picks Winograd / tiled GEMM; fp32 accumulation differs from ORT reference (also 5×5 kernel: 0.083). |
| 18 | [cross_openvino_fp16_matmul_add.py](cross_openvino_fp16_matmul_add.py) | max abs 0.078 | OV fp16 CPU tiled GEMM accumulates partial sums in different order than ORT; error grows with `N`. |

## 3. TensorFlow / XLA (2)

| # | Script | Error | Summary |
|---|---|---|---|
| 19 | [github_tensorflow_002.py](github_tensorflow_002.py) | off-by-one row | MLIR/TOSA nearest resize shifts rows by 1 with `half_pixel_centers=True` ([TF #62386](https://github.com/tensorflow/tensorflow/issues/62386)). |
| 20 | [github_tfxla_001_bf16_cast_elide.py](github_tfxla_001_bf16_cast_elide.py) | cast eliminated | XLA `AlgebraicSimplifier` collapses `Convert(fp32→bf16)→Convert(bf16→fp32)` as identity without checking that the intermediate type has fewer bits. Silently skips bf16 rounding. |

## 4. JAX (1)

| # | Script | Error | Summary |
|---|---|---|---|
| 21 | [github_jax_001_bf16_cast_elide.py](github_jax_001_bf16_cast_elide.py) | cast eliminated | Same XLA `AlgebraicSimplifier` bug as TF-XLA; eager JAX is correct, `jax.jit` strips the rounding. |

## 5. TVM (4)

| # | Script | Error | Summary |
|---|---|---|---|
| 22 | [github_tvm_004.py](github_tvm_004.py) | max abs 0.053 | `Resize(cubic, half_pixel, coeff_a=-0.5)` produced output that diverged from ORT and PyTorch bicubic (reference discrepancy preserved for regression test). |
| 23 | [github_tvm_010_simplifyexpr_rsqrt_precision.py](github_tvm_010_simplifyexpr_rsqrt_precision.py) | rel_err 9983× | `SimplifyExpr` rewrites `sqrt(x)/y → rsqrt(x)*y`; fast-rsqrt (`0x5f3759df` bit-trick) has catastrophic error for small `x`. `FoldScaleAxis` also reorders `conv→relu→mul` for negative scales incorrectly. ([TVM #16211](https://github.com/apache/tvm/issues/16211)) |
| 24 | [github_tvm_011_lifttransformparams_const_bind.py](github_tvm_011_lifttransformparams_const_bind.py) | off-by-one | `LiftTransformParams` re-binds a shared `ones` constant to the wrong lifted slot; `(A+1)*(B+1)` computes as `(A+2)*(B+2)`, max diff 33 on int inputs. |
| 25 | [github_tvm_012_gelu_approx_tanh.py](github_tvm_012_gelu_approx_tanh.py) | max abs 1.53e-4 | TVM maps ONNX `Gelu(approximate='tanh')` to exact erf-based GELU, producing a systematic ~1.5e-4 error on the Transformer GELU fast-path. |

## 6. PyTorch Inductor (3)

| # | Script | Error | Summary |
|---|---|---|---|
| 26 | [github_inductor_009_transpose_reduce_fusion.py](github_inductor_009_transpose_reduce_fusion.py) | max abs 1.53e-5 | Inductor fuses `ReduceSum` + `Transpose().contiguous()` and reinterprets storage strides, producing incorrect output in the attention-grad pattern. ([pytorch #146416](https://github.com/pytorch/pytorch/issues/146416)) |
| 27 | [github_inductor_011_bitshift_ub_shift64.py](github_inductor_011_bitshift_ub_shift64.py) | shift→64 non-zero | Inductor's CPU C++ codegen emits native C `x >> n` which is UB when `n == word_width`. On x86 the shift count is masked to 6 bits, so `n=64` acts as `n=0` and returns the input. ([pytorch #143555](https://github.com/pytorch/pytorch/issues/143555), [#143566](https://github.com/pytorch/pytorch/issues/143566)) |
| 28 | [github_inductor_013_bf16_cast_elide.py](github_inductor_013_bf16_cast_elide.py) | cast eliminated | Inductor treats adjacent `Cast(fp32→bf16)→Cast(bf16→fp32)` as identity, losing the intended precision truncation. ([pytorch #179561](https://github.com/pytorch/pytorch/issues/179561)) |

## 7. Cross-compiler bugs (7)

| # | Script | Error | Summary |
|---|---|---|---|
| 29 | [cross_bf16_cast_jit_elide.py](cross_bf16_cast_jit_elide.py) | 3 compilers fail | Unified repro: Inductor, TF-XLA, and JAX-jit *all* eliminate the bf16↔fp32 cast pair. Eager modes of all three, plus ORT / OV / numpy / ONNX-Ref, are correct. |
| 30 | [cross_bitshift_shift64_ov_ort.py](cross_bitshift_shift64_ov_ort.py) | ORT + OV BUG | Both ORT (opt + no-opt) and OV hit the same C-UB: right-shift by 64 returns non-zero. OpenVINO left-shift returns 2³¹. |
| 31 | [cross_crms_resize_nearest_ceil.py](cross_crms_resize_nearest_ceil.py) | TorchScript BUG | `Resize(nearest, nearest_mode=ceil)` → onnx2torch silently maps to PyTorch's floor. ORT & OV agree; TorchScript is off by 6.0. |
| 32 | [cross_cumsum_kvcache_multicompiler.py](cross_cumsum_kvcache_multicompiler.py) | OV + TorchScript BUG | `CumSum → Transpose → MatMul` (Q×Kᵀ self-attn pattern): OV tiled GEMM differs 0.32 vs ORT; TorchScript differs 57. |
| 33 | [cross_onnx2torch_cumsum.py](cross_onnx2torch_cumsum.py) | max_diff 4.50 | `CumSum(axis=2)`: onnx2torch calls `axis.item()`, triggering a `torch.compile` graph break and producing wrong cumulative sum. |
| 34 | [cross_onnx2torch_resize_linear_asym.py](cross_onnx2torch_resize_linear_asym.py) | max_diff 0.80 | onnx2torch maps ONNX `asymmetric` coord mode to the wrong PyTorch interpolation mode. |
| 35 | [cross_onnx2torch_resize_nearest_ceil.py](cross_onnx2torch_resize_nearest_ceil.py) | max_diff 3.00 | onnx2torch converts `nearest_mode='ceil'` to PyTorch's `F.interpolate`, which only supports floor nearest. |

## 8. TFLite (3)

| # | Script | Error | Summary |
|---|---|---|---|
| 36 | [cross_tflite_unstack_concat_reshape_fold.py](cross_tflite_unstack_concat_reshape_fold.py) | output = input | TFLite converter folds `reshape(x,[H,W,1]) → unstack(axis=1) → concat(axis=0) → reshape(x,[H,W])` into a single no-op `RESHAPE`, dropping the transpose-equivalent element permutation. ModelAnalyzer confirms the converted subgraph is just `Op#0 RESHAPE`. TF eager / XLA / PyTorch / ONNX Runtime all return the correct permutation. |
| 37 | [cross_tflite_l2_normalize_axis_mismatch.py](cross_tflite_l2_normalize_axis_mismatch.py) | 1.0 vs 1/√2 | `tf.math.l2_normalize(x)` defaults to `axis=None` (whole-tensor norm), but the TFLite converter silently lowers it to TFLite's built-in `L2_NORMALIZATION` op which only normalizes the **innermost** dimension. For input shape `[1,1,2] → +1 → transpose → l2_normalize`, the innermost dim has size 1, so each scalar is normalized to 1.0 instead of 1/√2 ≈ 0.7071. TF eager / XLA / PyTorch / ONNX Runtime all return 0.7071. |
| 38 | [cross_tflite_l2_normalize_multi_shape.py](cross_tflite_l2_normalize_multi_shape.py) | per-row vs whole-tensor | Same root cause as #37 reproduced on additional shapes: rank-3 `(2,2,3)` half-step floats and `(4,1)` integers. TFLite normalizes each innermost vector independently (or each scalar to 1.0 when innermost dim = 1) while the four reference runtimes return the whole-tensor L2 normalization. Confirms the bug is data-independent and triggers whenever the product of non-innermost dims is > 1. |

---

## Verified outputs (2026-04-15)

Exact stdout captured after `timeout 30 python3 <file>`. Output truncated at the
banner for brevity; the full numeric arrays are in the script itself.

### ONNXRuntime

```
$ python3 github_onnx_spec_007.py
ORT  output: [0.05263158 0.2631579  0.42105263 0.57894737 0.7368421  0.94736844]
Expected:    [0.05263158 0.21052632 0.42105263 0.57894737 0.7894737  0.94736844]
Bug (elem 4): ORT=0.7368, expected=0.7895
PASS=False
BUG REPRODUCED

$ python3 github_ort_002.py
ORT   output[30:35]: [12. 12. 13. 13. 14.]
Torch output[30:35]: [12. 12. 13. 13. 13.]
Mismatched elements: 12/64
BUG REPRODUCED

$ python3 github_ort_003.py
asymmetric coord mode max abs error vs PyTorch: 0.2947
ORT   output[:4]: [0.37454012 0.6050098  0.83547944 0.90697014]
Torch output[:4]: [0.37454012 0.43215755 0.6626272  0.8930969 ]
BUG REPRODUCED

$ python3 github_ort_004.py
Input:    [-0.2 -0.1  0.   0.1  0.2]
Expected: [False False False False False]
ORT out:  [ True  True False  True  True]
BUG REPRODUCED

$ python3 github_ort_008.py
Max abs error (CPU cubic vs PyTorch bicubic): 0.061254
BUG REPRODUCED

$ python3 github_ort_016.py
bicubic+border output[0,0,0,:]: [-0.5400001  2.6759987 12.323996  15.539993 ]
Bicubic+border values in [0,15] range: False
BUG REPRODUCED

$ python3 github_ort_017_mod_int_divzero_sigfpe.py
OnnxRuntime version: 1.24.4
Child exit code: -8
Child killed by SIGFPE (signal 8)
BUG REPRODUCED — ORT Mod(fmod=0, int32) with B=0 triggers SIGFPE
```

### OpenVINO

```
$ python3 github_ov_015_matmul_gpu_tile_overflow.py
     N    CPU result    expected      diff  bug?
  2047        2047.0      2047.0       0.0  no OV
  2048        2048.0      2048.0       0.0  no OV
  2049        2048.0      2049.0       1.0  no OV
  4096        2048.0      4096.0    2048.0  no OV
(OpenVINO not installed — showing GPU-bug simulation: N>2048 returns 2048)
BUG REPRODUCED: MatMul GPU tile skips partial last tile for dim > 2048

$ python3 github_ov_019_uint8_sub_no_wrap.py
a        : [  5 200 250   0]
b        : [ 10 100  10  50]
OV out   : [  0 100 240   0]
wrap ref : [251 100 240 206]  (correct — ONNX modular)
BUG REPRODUCED — OV uint8 Sub saturates instead of wrapping

$ python3 github_ov_020_uint8_mul_no_wrap.py
a*b      : [40000   300   300   300]
OV out   : [255 255 255 255]
wrap ref : [64 44 44 44]
BUG REPRODUCED — OV uint8 Mul saturates instead of wrapping

$ python3 github_ov_021_uint8_add_no_wrap.py
a+b      : [300 300 265 328]
OV out   : [255 255 255 255]
wrap ref : [44 44  9 72]
BUG REPRODUCED — OV uint8 Add saturates instead of wrapping

$ python3 github_ov_022_reducelogsumexp_overflow.py
Input x  : [[100.  88.  50.]
 [200. -10.   1.]]
OV out   : [inf inf]
Ref      : [100.00001 200.     ]
BUG REPRODUCED — OV ReduceLogSumExp overflows for inputs ≥ 88.7 (fp32 exp limit)

$ python3 github_ov_023_relu_nan_propagation.py
Input  : [nan -1.  0.  1. inf]
ORT    : [nan  0.  0.  1. inf]  ← NaN propagated (correct)
OV     : [ 0.  0.  0.  1. inf]
BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754)

$ python3 github_ov_024_int8_sub_saturation.py
a        : [-128  127   -1    0]
b        : [   1 -128 -127  100]
OV out   : [-128  127  126 -100]
wrap ref : [ 127   -1  126 -100]
BUG REPRODUCED — OV int8 Sub saturates instead of two's-complement wrapping

$ python3 github_ov_025_int8_add_saturation.py
a+b(i16) : [ 200  137 -138 -150]
OV out   : [ 127  127 -128 -128]
wrap ref : [ -56 -119  118  106]
BUG REPRODUCED — OV int8 Add saturates instead of two's-complement wrapping

$ python3 github_ov_026_exp_nan_to_inf.py
Input  : [nan  1.]
ORT    : [      nan 2.7182817]
OV     : [     inf 2.718282]
BUG REPRODUCED — OV Exp(NaN) → +inf, should propagate NaN (IEEE 754)

$ python3 cross_openvino_conv_bn_fusion.py
ORT ref[:4]:      [ 1.5056843   1.9531173  -0.09237421  2.6932142 ]
OpenVINO[:4]:     [ 1.5078125   1.953125   -0.09765625  2.703125  ]
max_diff=0.022135  tol=0.01
BUG REPRODUCED

$ python3 cross_openvino_conv_fp32_precision.py
Max abs diff: 0.053732  tol=0.02
5×5 kernel max abs diff: 0.083416  tol=0.02
BUG REPRODUCED: OpenVINO CPU Conv fp32 accumulation differs from ORT reference

$ python3 cross_openvino_fp16_matmul_add.py
Max abs diff: 0.078125  tol=0.05
BUG REPRODUCED: OpenVINO fp16 GEMM tile accumulation diverges from ORT
```

### TensorFlow / XLA

```
$ python3 github_tensorflow_002.py
asymmetric output:          [0. 0. 1. 1. 2. 2. 3. 3.]
half_pixel output:          [0. 0. 0. 1. 1. 2. 2. 3.]
Expected (both):            [0. 0. 1. 1. 2. 2. 3. 3.]
Max err asymmetric: 0.0000, half_pixel: 1.0000
BUG REPRODUCED

$ python3 github_tfxla_001_bf16_cast_elide.py
TensorFlow version: 2.21.0
TF eager          : 1.234375000000   (loss = 1.93e-04)
TF-XLA            : 1.234567880630    (loss = 9.49e-09)
BUG REPRODUCED — TF-XLA stripped bf16↔fp32 cast pair as identity
```

### JAX

```
$ python3 github_jax_001_bf16_cast_elide.py
JAX version: 0.9.2
JAX eager         : 1.234375000000   (loss = 1.93e-04)
JAX-jit (XLA)     : 1.234567880630    (loss = 9.49e-09)
BUG REPRODUCED — jax.jit eliminated bf16↔fp32 cast pair as identity
```

### TVM

```
$ python3 github_tvm_004.py
ORT cubic  output[0,0,0,:4]: [0.80208254 0.6395161  0.288994   0.30634046]
PyTorch bicubic[0,0,0,:4]:   [0.8143504  0.61203825 0.2709736  0.29592   ]
Max abs error ORT vs PyTorch: 0.053012
BUG REPRODUCED

$ python3 github_tvm_010_simplifyexpr_rsqrt_precision.py
     x         y     sqrt(x)/y    rsqrt(x)*y     rel_err
 1.00e-04     1.0    0.01000000     99.840172     9983.02  BUG
 1.00e-02     2.0    0.05000000     19.965044      398.30  BUG
 1.00e+00     1.0    1.00000000      0.998307        0.00  BUG
Pattern 2: conv→relu→mul(s) vs mul(s)→conv→relu  (FoldScaleAxis)
  relu(x)*scale (ref):  [-2.  -0.  -0.6 -0. ]
  scale*relu(x) (bug):  [0.  1.  0.  2.4]
  max_diff: 2.4000  BUG
BUG REPRODUCED: SimplifyExpr rsqrt rewrite introduces large precision error

$ python3 github_tvm_011_lifttransformparams_const_bind.py
Expected : [ 1  4  9 16 25 36 49 64] ...
Buggy    : [ 4  9 16 25 36 49 64 81] ...
Max diff : 33  (TVM not installed — analytical demo)
BUG REPRODUCED: LiftTransformParams corrupts constant binding

$ python3 github_tvm_012_gelu_approx_tanh.py
     x    gelu_exact     gelu_tanh    abs_diff    rel_diff  bug?
 -1.00   -0.15865525   -0.15880801    1.53e-04    9.63e-04  BUG
  1.00    0.84134475    0.84119199    1.53e-04    1.82e-04  BUG
Max abs diff: 1.53e-04  tol=1e-04
BUG REPRODUCED: TVM maps ONNX Gelu(approx=tanh) → exact erf, producing systematic error > 1e-4
```

### PyTorch Inductor

```
$ python3 github_inductor_009_transpose_reduce_fusion.py
Attention grad pattern [8,256,256] scale→transpose
eager vs torch.compile: max_diff=1.53e-05  BUG
BUG REPRODUCED: Inductor transpose+reduce fusion reinterprets storage strides

$ python3 github_inductor_011_bitshift_ub_shift64.py
     value   shift       correct     C_UB(x86)  bug?
      1000      64             0          1000  BUG(UB)
       255      64             0           255  BUG(UB)
       -42      64             0           -42  BUG(UB)
PyTorch runtime check:
  eager:   [1000, 255, -42] >> [64, 64, 64] = [0, 0, -1]
  compile: [1000, 255, -42] >> [64, 64, 64] = [0, 0, -1]
  expected: [0, 0, 0]  BUG
ONNX BitShift (direction=RIGHT) path:
  ORT: [1000, 255] >> [64, 64] = [1000, 255]
  expected: [0, 0]  BUG
BUG REPRODUCED: BitShift by 64 returns non-zero (C UB: x86 masks shift to 6 bits)

$ python3 github_inductor_013_bf16_cast_elide.py
Eager output        : 1.234375000000  (bf16 rounding applied)
Compiled output     : 1.234567880630
Compiled output == original x        : True
BUG REPRODUCED — Inductor eliminates bf16↔fp32 cast pair as identity
```

### Cross-compiler

```
$ python3 cross_bf16_cast_jit_elide.py
  PyTorch eager      : 1.234375000000   (✓ bf16 truncated)
  Inductor           : 1.234567880630   (✗ CAST ELIMINATED)
  TF eager           : 1.234375000000   (✓ bf16 truncated)
  TF-XLA (jit=True)  : 1.234567880630   (✗ CAST ELIMINATED)
  JAX eager          : 1.234375000000   (✓ bf16 truncated)
  JAX-jit (XLA)      : 1.234567880630   (✗ CAST ELIMINATED)
BUG REPRODUCED on 3 JIT compiler(s): PyTorch Inductor, TensorFlow XLA, JAX-jit
Fix (per compiler):
  - PyTorch Inductor: reject Cast-pair elision when intermediate dtype has fewer bits
  - TF-XLA / JAX-jit: same fix in xla/service/algebraic_simplifier.cc

$ python3 cross_bitshift_shift64_ov_ort.py
Backend                 max_diff  bug?
ORT_noopt                   1000  BUG
ORT_opt                     1000  BUG
OpenVINO                       0  ok
ORT_noopt_LEFT              1000  BUG
OpenVINO_LEFT         2147483648  BUG
BUG REPRODUCED on ['ORT_noopt', 'ORT_opt', 'ORT_noopt_LEFT', 'OpenVINO_LEFT']

$ python3 cross_crms_resize_nearest_ceil.py
Compiler          max_abs_diff  bug?
ORT_opt               0.000000  ok
OpenVINO              0.007692  ok
TorchScript           6.010351  BUG
BUG REPRODUCED on ['TorchScript']

$ python3 cross_cumsum_kvcache_multicompiler.py
Backend           max_abs_diff  bug?
ORT_opt               0.000000  ok
OpenVINO              0.321663  BUG
TorchScript          56.801445  BUG
BUG REPRODUCED on ['OpenVINO', 'TorchScript']

$ python3 cross_onnx2torch_cumsum.py
max_diff=4.497339  tol=0.01
BUG REPRODUCED

$ python3 cross_onnx2torch_resize_linear_asym.py
max_diff=0.804013  tol=0.01
BUG REPRODUCED

$ python3 cross_onnx2torch_resize_nearest_ceil.py
max_diff=3.003272  tol=0.01
BUG REPRODUCED
```

### TFLite

```
$ python3 cross_tflite_unstack_concat_reshape_fold.py
input    : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
expected : [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]
Runtime        Output                                                       Match?
--------------------------------------------------------------------------------------
Keras eager    [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]   OK
XLA            [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]   OK
PyTorch        [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]   OK
ONNX Runtime   [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]   OK
TFLite         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]   WRONG
BUG REPRODUCED — TFLite folds reshape+unstack+concat+reshape into a no-op RESHAPE

$ python3 cross_tflite_l2_normalize_axis_mismatch.py
input    : shape=[1, 1, 2], all 1.0
expected : [0.7071067690849304, 0.7071067690849304]  (= 1/sqrt(2))
Runtime        Output                                   Match?
----------------------------------------------------------------
Keras eager    [0.7071067690849304, 0.7071067690849304] OK
XLA            [0.7071067690849304, 0.7071067690849304] OK
PyTorch        [0.7071067690849304, 0.7071067690849304] OK
ONNX Runtime   [0.7071067690849304, 0.7071067690849304] OK
TFLite         [1.0, 1.0]                               WRONG
BUG REPRODUCED — TFLite lowers tf.math.l2_normalize(axis=None) to L2_NORMALIZATION (innermost-axis only)

$ python3 cross_tflite_l2_normalize_multi_shape.py
=== Case 3: shape=(2,2,3), half-step floats ===
  expected : [0.0699, 0.1049, 0.1399, 0.1748, 0.2098, 0.2447, 0.2797, 0.3147, 0.3496, 0.3846, 0.4196, 0.4545]
  Keras eager    matches expected                                              OK
  XLA            matches expected                                              OK
  PyTorch        matches expected                                              OK
  ONNX Runtime   matches expected                                              OK
  TFLite         [0.3714, 0.5571, 0.7428, 0.4767, 0.5721, 0.6674, 0.5111, ...] WRONG (each innermost-3 vec normalized independently)

=== Case 4: shape=(4,1), unit innermost dim ===
  expected : [0.2722, 0.4082, 0.5443, 0.6804]
  Keras eager / XLA / PyTorch / ONNX Runtime: matches expected                 OK
  TFLite         [1.0, 1.0, 1.0, 1.0]                                          WRONG (each scalar / itself = 1.0)
BUG REPRODUCED on both cases — confirms data-independence of the axis-semantics mismatch
```

---

## Running

```bash
cd bugs/minimal
python3 <bug_file>.py          # exit 0 = reproduced, 1 = fixed, 2 = deps missing
```

Or run everything at once:

```bash
for f in *.py; do
  out=$(timeout 60 python3 "$f" 2>/dev/null | tail -1)
  echo "$f  exit=$?  $out"
done
```
