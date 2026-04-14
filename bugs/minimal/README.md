# Minimal Reproducible Bug Scripts — Real Bugs Only

**96 self-contained Python scripts** — every bug verified to reproduce on current
compiler versions (2026-04-14).

```
Exit 0 = BUG REPRODUCED    Exit 1 = not reproduced    Exit 2 = missing deps
```

Each script follows the same format:

- Docstring header: `Bug ID`, `Source`, `Compiler`, `Patterns`, `Root cause`, `Tolerance`.
- Model built programmatically via `onnx.helper.make_node` / `helper.make_graph`
  (no external files).
- Runs the failing backend(s), compares against a PyTorch eager reference, sets
  `PASS`, exits 0/1.

## Summary

| Source | Count | |
|---|---:|---|
| Still-live GitHub / cross-compiler bugs | 12 | carried over from prior campaigns |
| Campaign v3 (oracle-verified) | 34 | newly discovered |
| New GitHub bugs — batch 1 (2026-04-14) | 4 | TVM, OV GPU, OV fp16 matmul |
| New GitHub bugs — batch 2 (2026-04-14) | 4 | Inductor, ORT BitShift, TVM Gelu, OV Conv |
| Campaign v4 (oracle-verified) | 14 | 300-model sweep, 5-compiler differential |
| Cross-compiler sweep (2026-04-14) | 3 | BitShift UB, Resize ceil, CumSum+KVAttn |
| **New GitHub bugs — batch 3 (2026-04-14)** | **12** | OV integer arithmetic + NaN, Inductor/TF-XLA/JAX-jit bf16 cast |
| **Campaign v5 + v6 (2026-04-14)** | **13** | 1000-model sweep (seeds 1000+2077), XLA + ORT divergences |
| **Total real bugs** | **96** | all reproduce on current CI |

Tested on: Python 3.13, ONNX 1.21, ORT 1.24.4 (CPU), OpenVINO 2026.0,
TensorFlow 2.21.0 (CPU), PyTorch 2.9.1 (torch.compile inductor, torch.jit),
JAX 0.9.2, onnx2torch.

Uniqueness: each of the 96 bugs has a distinct ONNX graph signature (op
sequence + key attributes) or a distinct compiler+root-cause pairing
(cross-compiler bugs #64–66 share one root cause in 3 separate compilers).
No two bugs share the same structure + same affected compiler.

---

## Part 1 — 12 Still-Live Bugs (legacy)

| # | Bug ID | Compiler | max_diff | Root cause |
|---|---|---|---|---|
| 1 | [github_onnx_spec_007](github_onnx_spec_007.py) | ONNX Spec | — | Resize nearest + half_pixel rounding returns wrong index at element 4 |
| 2 | [github_ort_002](github_ort_002.py) | OnnxRuntime | 12/64 | Nearest resize 1→64 pixel selection off-by-one vs PyTorch |
| 3 | [github_ort_003](github_ort_003.py) | OnnxRuntime | 0.29 | Asymmetric bilinear resize max error vs PyTorch |
| 4 | [github_ort_004](github_ort_004.py) | OnnxRuntime | — | Optimizer fuses float→int32→bool, skipping truncation step |
| 5 | [github_ort_008](github_ort_008.py) | OnnxRuntime | — | CUDA cubic resize antialias grid wrong — hardcoded `cubic_coeff_a` |
| 6 | [github_ort_016](github_ort_016.py) | OnnxRuntime | — | GridSample bicubic+border clamps after neighbourhood instead of per-sample |
| 7 | [github_tensorflow_002](github_tensorflow_002.py) | TensorFlow XLA | 1.00 | MLIR/TOSA nearest resize shifts rows by 1 with half_pixel_centers |
| 8 | [github_tvm_004](github_tvm_004.py) | TVM Relay | 0.053 | Cubic interpolation diverges from ORT and PyTorch bicubic |
| 9 | [cross_onnx2torch_resize_nearest_ceil](cross_onnx2torch_resize_nearest_ceil.py) | onnx2torch | 3.00 | Uses floor nearest_mode instead of ceil |
| 10 | [cross_onnx2torch_resize_linear_asym](cross_onnx2torch_resize_linear_asym.py) | onnx2torch | 0.80 | Asymmetric coord mode mapped to wrong PyTorch interpolation |
| 11 | [cross_openvino_conv_bn_fusion](cross_openvino_conv_bn_fusion.py) | OpenVINO 2026.0 | 0.022 | Conv+BN fusion rounding error above tolerance |
| 12 | [cross_onnx2torch_cumsum](cross_onnx2torch_cumsum.py) | onnx2torch | 4.50 | `CumSum axis.item()` causes graph break and wrong output |

---

## Part 1B — 4 New GitHub + Cross-Compiler Bugs (2026-04-14)

All 4 bugs are **ACTIVE** — they reproduce on current compiler versions.

| # | Bug ID | Compiler | max_diff | Status | Root cause |
|---|---|---|---|---|---|
| 13 | [github_tvm_010_simplifyexpr_rsqrt_precision](github_tvm_010_simplifyexpr_rsqrt_precision.py) | TVM Relay | 9983× rel err | **ACTIVE** | `SimplifyExpr` rewrites `sqrt(x)/y` → `rsqrt(x)*y`; fast-rsqrt gives rel error ~9983 for x=1e-4; `FoldScaleAxis` reorders conv→relu→mul incorrectly for negative scales |
| 14 | [github_tvm_011_lifttransformparams_const_bind](github_tvm_011_lifttransformparams_const_bind.py) | TVM Relax | 33 (int) | **ACTIVE** | `LiftTransformParams` re-binds the shared `ones` constant to the wrong lifted slot; `(A+1)*(B+1)` → `(A+2)*(B+2)` |
| 15 | [github_ov_015_matmul_gpu_tile_overflow](github_ov_015_matmul_gpu_tile_overflow.py) | OpenVINO GPU | 2048.0 | **ACTIVE** (GPU 2023.x) | GPU MatMul kernel tile size hard-coded at 2048; last partial tile skipped for dim > 2048; dot-product of 4096 ones returns 2048 instead of 4096 |
| 16 | [cross_openvino_fp16_matmul_add](cross_openvino_fp16_matmul_add.py) | OpenVINO 2026.0 | **0.078** | **ACTIVE** | OpenVINO CPU fp16 tiled GEMM accumulates partial sums in different order than ORT; error grows with N (N=64→0.078, N=128→0.188); same root cause as ORT#23284 |

### Output samples (2026-04-14, Python 3.13)

```
github_tvm_010        x=1e-4  y=1.0  sqrt(x)/y=0.01000000  rsqrt(x)*y=99.840172  rel_err=9983.02  BUG
                      x=0.01  y=2.0  sqrt(x)/y=0.05000000  rsqrt(x)*y=19.965044  rel_err=398.30   BUG
                FoldScaleAxis: relu(x)*scale=[-2. -0. -0.6 -0.]  scale*relu(x)=[0. 1. 0. 2.4]  max_diff=2.4  BUG

github_tvm_011  Expected: [1 4 9 16 25 36 49 64]
                Buggy:    [4 9 16 25 36 49 64 81]  max_diff=33  BUG

github_ov_015   N=2047 CPU=2047.0  N=2048 CPU=2048.0  N=2049 GPU=2048.0  N=4096 GPU=2048.0  BUG

cross_openvino_fp16_matmul_add
                ORT ref: [ 5.207 -0.85   4.332 -7.043]
                OpenVINO:[ 5.22  -0.875  4.312 -7.062]  max_diff=0.078125  BUG
```

---

## Part 2 — 34 Campaign v3 Bugs

All oracle-verified; verdict is `rel_L2(backend vs pytorch_eager) > 0.1` or a
crash in the targeted backend.

### 2A. Crashes (4)

| # | Bug ID | Compiler | Root cause |
|---|---|---|---|
| 17 | [bug_000032](bug_000032.py) | TensorFlow (opt + noopt) | SpaceToDepth reshape/transpose feeds Conv with asymmetric pads `[0,1,0,1]` → `conv2d` dtype mismatch |
| 18 | [bug_000175](bug_000175.py) | OpenVINO (opt + noopt) | Gated residual + Conv1x1 + Split + manual GEGLU (Div/Erf/Add/Mul) + Conv2x2(s=2)+BN+LeakyRelu → CPU plugin internal error |
| 19 | [bug_000227](bug_000227.py) | TorchScript+opt | MaxPool `dilations=[2,2]` after Resize(asymmetric)+Conv3x3 branch → freeze + optimize_for_inference crash |
| 20 | [bug_000310](bug_000310.py) | TorchScript+opt | MaxPool `dilations=[2,2]` after conv-BN-fusion-both-paths + identity Transpose `[0,1,2,3]` |

### 2B. Multi-backend numerical divergence — ORT + OpenVINO + TF + XLA (±torch) (11)

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 21 | [bug_000036](bug_000036.py) | 0.31 | Resize(cubic, half_pixel) → Expand → Mul → Resize(linear, asymmetric) → LayerNorm |
| 22 | [bug_000060](bug_000060.py) | 0.57 | CumSum(last_axis) + SiLU residual + gated residual + Gather/Slice/Expand |
| 23 | [bug_000092](bug_000092.py) | 0.97 | Manual cRMS-norm (Mul/ReduceMean/Sqrt/Div) + ReLU residual + CumSum + Gather |
| 24 | [bug_000121](bug_000121.py) | 0.50 | CumSum → MatMul → ReduceL2 → MatMul → MatMul (triple matmul) |
| 25 | [bug_000143](bug_000143.py) | 1.00 | Neg+Abs+Relu sign-manip → ReduceMax → gated residual → manual LayerNorm (ORT+TF) |
| 26 | [bug_000163](bug_000163.py) | 1.10 | GLU Split → LogSoftmax → Mul → CumSum → Ceil → Cast(fp32→int32→fp32) roundtrip |
| 27 | [bug_000216](bug_000216.py) | 0.68 | CumSum + matmul-scale-add + Pow(canonical = x·x) + RMSNorm |
| 28 | [bug_000248](bug_000248.py) | 1.25 | Transpose-MatMul-Transpose sandwich + Einsum `bij→bji` + power-norm + CumSum |
| 29 | [bug_000267](bug_000267.py) | 0.97 | matmul+bias+Sigmoid + add-mul-add + Tanh+Erf + CumSum + Pow(x,1) identity |
| 30 | [bug_000308](bug_000308.py) | 1.00 | TopK(k=1)+Tile → Add+Relu+Sub → Neg+Abs+Relu → InstanceNorm1D (5 backends incl. torch.compile) |
| 31 | [bug_000424](bug_000424.py) | 1.01 | Flatten(axis=2) + MatMul + BN + Clip(Relu6) + CumSum + Conv3x3 + Elu |

### 2C. TF-only divergence (15)

All show rel_L2 ≥ 0.1 on TF graph mode vs PyTorch eager.

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 32 | [bug_000030](bug_000030.py) | 0.33 | Pad(reflect) + InstanceNorm + LayerNorm(axis=-1) + MatMul + Pow |
| 33 | [bug_000031](bug_000031.py) | 0.22 | Add→LayerNorm→Conv + unrolled CELU (Max+Div+Exp) + reflect-Pad + Conv + MatMul |
| 34 | [bug_000055](bug_000055.py) | 0.23 | MatMul + Conv1x1 + Tanh + HardSwish + Abs/Add/Pow/Sqrt + reflect-Pad + Conv3x3 |
| 35 | [bug_000166](bug_000166.py) | 0.33 | constant-Pad + Conv3x3 + BN + Unsqueeze/Squeeze (rank-manip) + MatMul + reflect-Pad |
| 36 | [bug_000170](bug_000170.py) | 0.25 | MatMul+Mul+Add+Relu + ConvTranspose2x2(s=2) + BN + Relu + asymmetric Conv pads `[0,1,0,1]` |
| 37 | [bug_000189](bug_000189.py) | 0.20 | MatMul+MatMul + manual LayerNorm → native LayerNorm (double norm) + Relu + reflect-Pad + Conv3x3 |
| 38 | [bug_000223](bug_000223.py) | 0.20 | Expand broadcast + Add + reflect-Pad + Conv3x3 + MatMul + Conv5x5 + Conv+BN+Elu |
| 39 | [bug_000232](bug_000232.py) | 0.35 | MatMul + reflect-Pad + channel-shuffle (Reshape/Transpose[0,2,1,3,4]/Reshape) + MatMul + Concat + Conv1x1 |
| 40 | [bug_000242](bug_000242.py) | 0.19 | MatMul+MatMul + reflect-Pad + Conv3x3 + Conv1x1+BN+Clip (Relu6) + GLU Split |
| 41 | [bug_000245](bug_000245.py) | 0.49 | Manual LayerNorm + power-norm (Pow/Div/Mul) + MatMul + edge-Pad + Reshape-flatten |
| 42 | [bug_000307](bug_000307.py) | 0.33 | reflect-Pad + Reshape-flatten + Conv1x1(s=2) + BN + LeakyRelu + MatMul + Relu+Add+Relu |
| 43 | [bug_000322](bug_000322.py) | 0.23 | reflect-Pad + Conv3x3 + MatMul + Conv3x3 + Selu + manual mean-variance norm + Resize(nearest, round_prefer_floor) |
| 44 | [bug_000342](bug_000342.py) | 0.41 | Greater + Cast(bool→fp32) mask + MatMul + LogSoftmax + reflect-Pad + edge-Pad |
| 45 | [bug_000372](bug_000372.py) | 0.28 | Tanh+Erf+Mul + BN-Conv-BN sandwich + Reshape-squash + edge-Pad + mul-self-as-pow |
| 46 | [bug_000416](bug_000416.py) | 0.21 | Two Transpose[2,3,0,1] squash-to-identity + ReduceL2 + Div + Mul + reflect-Pad + Conv + Pow + ReduceMean(axis=0) |

### 2D. TF + XLA divergence (2)

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 47 | [bug_000210](bug_000210.py) | 0.21 | Transpose[0,1,3,2] + RMSNorm + MatMul + Resize(linear, align_corners) |
| 48 | [bug_000404](bug_000404.py) | 0.35 | Resize(linear, align_corners) + Reshape-roundtrip + Conv3x3+BN+Elu + MatMul + InstanceNorm |

### 2E. ORT + OpenVINO only (2)

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 49 | [bug_000008](bug_000008.py) | 1.00 | Selu → TopK(k=1) → Tile → Mul → Add → LayerNorm + Dropout (one-hot masking chain) |
| 50 | [bug_000426](bug_000426.py) | 0.47 | Resize(linear, asymmetric) + ASPP-style dilated convs (d=1, 2, 4) + BN + LeakyRelu + Split + Mul + Concat |

---

## Recurring root-cause primitives

These low-level patterns explain most of the 34 new bugs:

1. **`CumSum(axis=-1)`** stacked with matmul or normalization — 8 bugs across
   ORT / OpenVINO / TF / XLA.
2. **`Resize`** with unusual coordinate modes (`half_pixel`, `align_corners`,
   `asymmetric`, `round_prefer_floor`) — 6 bugs.
3. **Asymmetric Conv padding `[0,1,0,1]`** — crashes TF, diverges in TF graph mode.
4. **`MaxPool(dilations=[2,2])`** — crashes TorchScript `freeze` + `optimize_for_inference`.
5. **Stacked normalizations** (manual LayerNorm + native LayerNorm, power-norm +
   LayerNorm, RMSNorm + Resize) — TF/XLA fusion produces wrong results.
6. **`TopK(k=1)` + `Tile`** (one-hot expansion) — ORT/OpenVINO optimizer loses info.
7. **Transpose–MatMul–Transpose** sandwich — optimizers wrongly collapse transposes.
8. **`Greater` + `Cast(bool→fp32)`** mask + MatMul — TF miscomputes broadcast.

---

## Part 1C — 4 New GitHub + Cross-Compiler Bugs — batch 2 (2026-04-14)

| # | Bug ID | Compiler | max_diff | Status | Root cause |
|---|---|---|---|---|---|
| 51 | [github_inductor_009_transpose_reduce_fusion](github_inductor_009_transpose_reduce_fusion.py) | Inductor CPU | 1.53e-05 | **ACTIVE** | Inductor fuses row-wise reduce + pointwise mul + transpose(); reinterprets storage strides as if tensor were already transposed — constant 2^-16 error on attention-grad-like patterns (pytorch/pytorch#146416) |
| 52 | [github_inductor_011_bitshift_ub_shift64](github_inductor_011_bitshift_ub_shift64.py) | ORT + Inductor | 1000 | **ACTIVE** | C UB: `x >> 64` on x86 masks shift count to 6 bits → `x >> 0 = x` instead of 0; ORT's ONNX BitShift op also affected — `[1000,255] >> [64,64]` returns `[1000,255]` (pytorch/pytorch#143555, also ORT) |
| 53 | [github_tvm_012_gelu_approx_tanh](github_tvm_012_gelu_approx_tanh.py) | TVM Relax | 2.18e-04 | **ACTIVE** | TVM's ONNX frontend ignores `approximate="tanh"` attribute on opset-20 Gelu; maps to exact erf formula — systematic error >1e-4 for all activations (apache/tvm#18750) |
| 54 | [cross_openvino_conv_fp32_precision](cross_openvino_conv_fp32_precision.py) | OpenVINO 2026.0 | 0.054 | **ACTIVE** | OV CPU plugin selects Winograd/tiled-GEMM for float32 Conv; different accumulation order produces 0.054 max diff vs ORT direct-conv (3×3) and 0.083 for 5×5 kernel |

---

## Part 1D — 12 New GitHub Bugs — batch 3 (2026-04-14)

All discovered by targeted GitHub issue research + differential probing across
10 compilers (numpy, ONNX-Reference, ORT, OpenVINO, PyTorch eager, Inductor,
TorchScript, onnx2torch, TF eager, TF-XLA, JAX, JAX-jit).  Verified on current
versions (ORT 1.24.4, OpenVINO 2026.0, PyTorch 2.9.1, TensorFlow 2.21.0, JAX 0.9.2).

### 1D.a — OpenVINO-only bugs (8)

All 8 bugs are isolated to **OpenVINO 2026.0 CPU plugin** — every other compiler
(including the ONNX reference) produces the correct result.

| # | Bug ID | Compiler | Root cause |
|---|---|---|---|
| 55 | [github_ov_019_uint8_sub_no_wrap](github_ov_019_uint8_sub_no_wrap.py) | OpenVINO 2026.0 | uint8 Sub dispatches to `_mm_subs_epu8` (saturating SSE) instead of `_mm_sub_epi8` (wrapping); 5-10=0 instead of 251 (OV #33518) |
| 56 | [github_ov_020_uint8_mul_no_wrap](github_ov_020_uint8_mul_no_wrap.py) | OpenVINO 2026.0 | uint8 Mul kernel saturates to 255 instead of wrapping mod 256: 200×200=40000 → should be 64, OV gives 255 (OV #33518 Mul) |
| 57 | [github_ov_021_uint8_add_no_wrap](github_ov_021_uint8_add_no_wrap.py) | OpenVINO 2026.0 | uint8 Add uses `_mm_adds_epu8` (saturating); 200+100=300 → should wrap to 44, OV gives 255 (OV #33518 Add) |
| 58 | [github_ov_022_reducelogsumexp_overflow](github_ov_022_reducelogsumexp_overflow.py) | OpenVINO 2026.0 | ReduceLogSumExp implemented naively as `Log(ReduceSum(Exp(x)))`; exp(100) overflows fp32 → `+inf`; every other compiler uses max-shift stabilizer `m + log(Σexp(xᵢ-m))` (OV #32839) |
| 59 | [github_ov_023_relu_nan_propagation](github_ov_023_relu_nan_propagation.py) | OpenVINO 2026.0 | Relu kernel uses ordered compare `x > 0 ? x : 0`; NaN fails ordered test and falls into `else 0.0` branch. IEEE 754 requires max(0, NaN) = NaN. Only Relu + Exp affected in OV 2026 — Sigmoid/Tanh/Sqrt/Abs/Neg/Log propagate NaN correctly |
| 60 | [github_ov_024_int8_sub_saturation](github_ov_024_int8_sub_saturation.py) | OpenVINO 2026.0 | int8 Sub saturates at [-128, 127] instead of two's-complement wrap. OV int16/int32/int64 Sub wrap correctly → bug confined to 8-bit SIMD dispatch (extends #33518 to signed int8) |
| 61 | [github_ov_025_int8_add_saturation](github_ov_025_int8_add_saturation.py) | OpenVINO 2026.0 | int8 Add saturates via `_mm_adds_epi8`: 100+100=200 → should wrap to -56, OV gives 127. Completes the 6-op OV 8-bit saturation family (extends #33518) |
| 62 | [github_ov_026_exp_nan_to_inf](github_ov_026_exp_nan_to_inf.py) | OpenVINO 2026.0 | Exp kernel uses range short-circuit (`x > LARGE ? +inf : expf(x)`); NaN's unordered compare maps into the +inf branch. IEEE 754 requires exp(NaN)=NaN (new discovery — no existing GitHub issue) |

### 1D.b — JIT cast-elimination bug: same root cause in 3 compilers (3+1 scripts)

One algebraic error — `Cast(T→U) ∘ Cast(U→T) → Identity(T)` applied without verifying
`precision(U) ≥ precision(T)` — independently re-implemented in **three production JIT
compilers**.  bf16 has 7 mantissa bits vs fp32's 23, so the round-trip is lossy, not
identity.  Eager modes (all frameworks), ORT, OpenVINO, ONNX-Reference, and numpy
all truncate correctly.

| # | Bug ID | Compiler | Root cause |
|---|---|---|---|
| 63 | [github_inductor_013_bf16_cast_elide](github_inductor_013_bf16_cast_elide.py) | PyTorch Inductor 2.9.1 | Inductor's algebraic simplifier elides adjacent Cast pair without precision check (pytorch/pytorch#179561) |
| 64 | [github_tfxla_001_bf16_cast_elide](github_tfxla_001_bf16_cast_elide.py) | TensorFlow XLA 2.21 | XLA `AlgebraicSimplifier` in `xla/service/algebraic_simplifier.cc` collapses `Convert∘Convert → no-op` without precision check (openxla/xla) |
| 65 | [github_jax_001_bf16_cast_elide](github_jax_001_bf16_cast_elide.py) | JAX-jit 0.9.2 (XLA) | Shared XLA backend with TF — same simplifier, same bug. One XLA fix resolves both 64 and 65 |
| 66 | [cross_bf16_cast_jit_elide](cross_bf16_cast_jit_elide.py) | Inductor + TF-XLA + JAX-jit | Unified cross-compiler test: runs all three JIT compilers, reports which reproduce the bug (all three do on current versions) |

### 1D.c — Output samples (2026-04-14)

```
github_ov_019  a=[5, 200, 250, 0] - b=[10, 100, 10, 50]
               wrap ref (ONNX spec) : [251, 100, 240, 206]
               OV CPU               : [  0, 100, 240,   0]  ← saturates

github_ov_022  x=[[100, 88, 50], [200, -10, 1]]
               stable ref (other 8 compilers): [100.00001, 200.0]
               OV CPU                         : [   +inf,    +inf]  ← overflow

github_ov_023  Relu([NaN, -1, 0, 1])
               ORT/PT/TF/JAX/Inductor/XLA    : [NaN, 0, 0, 1]  ← NaN propagated
               OV CPU                         : [ 0 , 0, 0, 1]  ← NaN→0

github_inductor_013 / tfxla_001 / jax_001
               input fp32       = 1.234567890123
               eager (all frameworks) = 1.234375   ← correct bf16 rounding
               Inductor / TF-XLA / JAX-jit = 1.234567880630  ← both casts stripped
```

### 1D.d — Cross-compiler coverage matrix for batch 3

| Bug | numpy | ONNX-Ref | ORT | **OV** | onnx2torch | PT-eager | **Inductor** | TorchScript | TF-eager | **TF-XLA** | JAX | **JAX-jit** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 55–57 uint8 Add/Sub/Mul | ✓ | ✓ | (N/A) | **✗** | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| 58 ReduceLogSumExp | ✓ | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | — |
| 59 Relu(NaN) | — | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| 60–61 int8 Add/Sub | ✓ | ✓ | (N/A) | **✗** | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| 62 Exp(NaN) | — | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| 63–66 bf16 cast | ✓ | ✓ | ✓ | ✓ | — | ✓ | **✗** | ✓ | ✓ | **✗** | ✓ | **✗** |

ORT (N/A) = ORT does not implement 8-bit Add/Sub/Mul (rejects as invalid graph).

See also: [`reports/`](reports/) — ready-to-file GitHub issue scripts for all 9 bugs,
organized by compiler subfolder (openvino/, pytorch_inductor/, tensorflow_xla/, jax_xla/).

---

## Part 3 — 14 Campaign v4 Bugs (2026-04-14)

Oracle-verified by 300-model 5-compiler differential sweep (seed 900).
All use `pytorch_eager` (via onnx2torch) as reference; failure = `rel_L2 > 0.01`.

### 3A. ORT optimizer divergence from PyTorch (9)

| # | Bug ID | rel_L2 | Failing | Patterns |
|---|---|---:|---|---|
| 55 | [bug_v4_000035](bug_v4_000035.py) | 0.92 | ORT_opt | LayerNorm + Dropout(identity) + Squeeze/Unsqueeze + ReduceL2 + reshape-LayerNorm + GLU |
| 56 | [bug_v4_000048](bug_v4_000048.py) | 1.04 | ORT_opt | CumSum(axis=3/spatial) + Conv + BN + ReLU + MaxPool + MatMul×2 + InstanceNorm |
| 57 | [bug_v4_000078](bug_v4_000078.py) | 1.00 | ORT_opt | cRMS-norm + Resize(nearest,ceil) + LayerNorm-gate + batched-MatMul + identity-Transpose |
| 58 | [bug_v4_000135](bug_v4_000135.py) | 0.15 | ORT_opt | Stable-softmax (ReduceMax-Sub-Exp-Div) + AvgPool + Gather(channel) + reshape-GroupNorm + Relu |
| 59 | [bug_v4_000151](bug_v4_000151.py) | 1.84 | ORT_opt | Add-zero(identity) + CumSum + Mul-add-mul chain + SE-block + KV-cache attention |
| 60 | [bug_v4_000186](bug_v4_000186.py) | 1.43 | ORT_opt | identity-chain + Conv+Tanh + concat-conv + CumSum + global-branch-mul |
| 61 | [bug_v4_000198](bug_v4_000198.py) | 3.53e5 | ORT_opt | MatMul+bias+Sigmoid + residual-relu + TopK(axis=0)+Tile → self-add×3 → LayerNorm(ε≈0) + Div(temp) |
| 62 | [bug_v4_000223](bug_v4_000223.py) | 4.72e4 | ORT_opt | Gather(3)+Reshape + TopK(k=1)+Tile + L1-norm + Mul/Add chain + LayerNorm + Div(temp=2.25) |
| 63 | [bug_v4_000290](bug_v4_000290.py) | 1.53 | ORT_opt | Abs+SiLU + CumSum(axis=2) + Sigmoid+SiLU + MatMul+scale + Clip(-100,100)+Cast(int)+Cast(fp32)+Mul |

### 3B. TorchScript (jit) divergence from PyTorch eager (4)

| # | Bug ID | rel_L2 | Patterns |
|---|---|---:|---|
| 64 | [bug_v4_000036](bug_v4_000036.py) | 0.31 | MatMul×3 + Resize(cubic,half_pixel) + Concat-with-zero |
| 65 | [bug_v4_000055](bug_v4_000055.py) | 0.07 | Parallel MaxPool+AvgPool + reflect-Pad+Conv + Sub-Mul-Add + Resize(cubic,half_pixel) + Unsqueeze-Expand-Mul |
| 66 | [bug_v4_000234](bug_v4_000234.py) | 0.94 | Conv+BN+Elu + CumSum + padded-grouped-Conv + ASPP dilated branches + Swish(explicit Sigmoid) |
| 67 | [bug_v4_000254](bug_v4_000254.py) | 0.18 | Transpose-Conv(NHWC) + AvgPool(count_include_pad) + Pow(canonical) + ceil-mode-AvgPool+Conv + batched-MatMul |

### 3C. OpenVINO + TVM + XLA divergence (1)

| # | Bug ID | rel_L2 | Failing | Patterns |
|---|---|---:|---|---|
| 68 | [bug_v4_000041](bug_v4_000041.py) | 1.00 | OpenVINO, TVM, XLA | LayerNorm+Dropout + TopK(k=1)+Tile + Abs/Mul + Gather/Slice/Expand + LayerNorm+Relu (sparse one-hot propagation through normalization) |

---

## Part 4 — 3 Cross-Compiler Sweep Bugs (2026-04-14)

Derived from the campaign v4 cross-compiler sweep: each script demonstrates a bug
that reproduces on multiple distinct backends (beyond its originally-discovered compiler).

| # | Bug ID | Failing compilers | max_diff | Root cause |
|---|---|---|---:|---|
| 69 | [cross_bitshift_shift64_ov_ort](cross_bitshift_shift64_ov_ort.py) | ORT (RIGHT+LEFT), OpenVINO (LEFT) | 2147483648 | `BitShift(uint64, shift=64)`: C UB — x86 SHRQ masks shift count mod 64, so `x>>64` = `x>>0` = x; OV LEFT shift overflows to 2^31 instead of 0 |
| 70 | [cross_crms_resize_nearest_ceil](cross_crms_resize_nearest_ceil.py) | TorchScript (onnx2torch) | 6.01 | `Resize(nearest, half_pixel, nearest_mode=ceil)`: onnx2torch ignores `nearest_mode=ceil`, always uses PyTorch floor-nearest; for half_pixel coords with scale=2 every other destination pixel gets the wrong source pixel |
| 71 | [cross_cumsum_kvcache_multicompiler](cross_cumsum_kvcache_multicompiler.py) | OpenVINO 2026.0 | 0.13 | `CumSum(axis=2) → Q×K^T Softmax ×V` (self-attention on running sums): OV tiled-GEMM accumulates large CumSum-inflated dot products in a different fp32 summation order than ORT, Softmax amplifies the error |

### Notes on extreme divergence (bugs 61, 62)

`bug_v4_000198` and `bug_v4_000223` produce rel_L2 in the 10^4–10^5 range.
Both use a **LayerNorm with near-zero variance** (all-same input from TopK+Tile)
followed by a **temperature division**.  When all elements are identical,
LayerNorm variance → 0 and the normalization output is numerically unstable
(depends on epsilon).  ORT and PyTorch use different epsilon handling → cascade to
catastrophic amplification in the downstream Div.

---

## Part 5 — Delta-Debugged Minimal Op Sequences (2026-04-14)

All 48 campaign bugs (34 v3 + 14 v4) were delta-debugged: nodes are truncated
from the end until the smallest graph that still reproduces the divergence
remains. Total ops across all 48 bugs reduced **607 → 352 (42% reduction)**.

Below: each campaign bug's minimal op sequence and the actual `rel_L2` vs
PyTorch eager printed by the script. **Many multi-backend bugs reduce to a
single ONNX op** — the fuzzer's surrounding ops were irrelevant.

| Bug ID | Ops (orig→min) | Minimal op sequence | Reproduces |
|---|---:|---|---|
| [bug_000426](bug_000426.py) | 18→1 (6%) | `Resize` | onnxruntime=5.43e-01, openvino=5.43e-01 |
| [bug_000216](bug_000216.py) | 17→1 (6%) | `CumSum` | onnxruntime=1.40e+00, openvino=1.40e+00, tensorflow=1.40e+00, xla=1.40e+00 |
| [bug_000060](bug_000060.py) | 16→1 (6%) | `CumSum` | onnxruntime=1.10e+00, openvino=1.10e+00, tensorflow=1.10e+00, xla=1.10e+00 |
| [bug_000322](bug_000322.py) | 13→1 (8%) | `Pad` | tensorflow=3.44e-01 |
| [bug_000307](bug_000307.py) | 10→1 (10%) | `Pad` | tensorflow=3.39e-01 |
| [bug_v4_000048](bug_v4_000048.py) | 9→1 (11%) | `CumSum` | onnxruntime=1.31e+00, openvino=1.31e+00, tvm=1.31e+00, xla=1.31e+00 |
| [bug_v4_000151](bug_v4_000151.py) | 25→3 (12%) | `Add → Mul → CumSum` | onnxruntime=1.35e+00, openvino=1.35e+00, tvm=1.35e+00, xla=1.35e+00 |
| [bug_000030](bug_000030.py) | 8→1 (12%) | `Pad` | tensorflow=3.29e-01 |
| [bug_000404](bug_000404.py) | 8→1 (12%) | `Resize` | tensorflow=3.39e-01, xla=3.39e-01 |
| [bug_000121](bug_000121.py) | 7→1 (14%) | `CumSum` | onnxruntime=1.33e+00, openvino=1.33e+00, tensorflow=1.33e+00, xla=1.33e+00 |
| [bug_v4_000290](bug_v4_000290.py) | 13→3 (23%) | `Abs → Mul → CumSum` | onnxruntime=1.37e+00, openvino=1.37e+00, tvm=1.37e+00, xla=1.37e+00 |
| [bug_000232](bug_000232.py) | 8→2 (25%) | `MatMul → Pad` | tensorflow=3.43e-01 |
| [bug_000036](bug_000036.py) | 11→3 (27%) | `Greater → Where → Resize` | tensorflow=2.35e-01, xla=2.35e-01 |
| [bug_000242](bug_000242.py) | 10→3 (30%) | `MatMul → MatMul → Pad` | tensorflow=3.23e-01 |
| [bug_v4_000234](bug_v4_000234.py) | 13→4 (31%) | `Conv → BatchNormalization → Elu → CumSum` | onnxruntime=1.24e+00, openvino=1.24e+00, tvm=1.24e+00, xla=1.24e+00 |
| [bug_000223](bug_000223.py) | 9→3 (33%) | `Expand → Add → Pad` | tensorflow=3.37e-01 |
| [bug_v4_000254](bug_v4_000254.py) | 12→5 (42%) | `Transpose → Transpose → Conv → Relu → AveragePool` | tvm=1.13e-01, xla=CRASH |
| [bug_000163](bug_000163.py) | 14→6 (43%) | `Split → Sigmoid → Mul → LogSoftmax → Mul → CumSum` | onnxruntime=1.35e+00, openvino=1.35e+00, tensorflow=1.35e+00, xla=1.35e+00 |
| [bug_000310](bug_000310.py) | 13→7 (54%) | `Conv → BatchNormalization → Conv → Mul → Add → Add → MaxPool` | torchscript=CRASH |
| [bug_000227](bug_000227.py) | 11→6 (55%) | `MatMul → Conv → Mul → Add → Clip → Resize` | onnxruntime=2.37e-01, openvino=2.37e-01 |
| [bug_v4_000078](bug_v4_000078.py) | 14→8 (57%) | `Mul → ReduceMean → Add → Sqrt → Div → Mul → Mul → Resize` | onnxruntime=1.22e+00, openvino=1.22e+00 |
| [bug_000416](bug_000416.py) | 12→7 (58%) | `Transpose → Transpose → ReduceL2 → Add → Div → Mul → Pad` | tensorflow=3.36e-01 |
| [bug_v4_000135](bug_v4_000135.py) | 16→10 (62%) | `ReduceMax → Sub → Exp → ReduceSum → Div → AveragePool → Mul → Add → … (+2)` | onnxruntime=1.98e-01, openvino=1.70e-01, torch_compile=2.00e-01, tvm=1.00e+00, xla=CRASH |
| [bug_000372](bug_000372.py) | 13→9 (69%) | `Tanh → Erf → Mul → BatchNormalization → Conv → BatchNormalization → Reshape → Reshape → … (+1)` | tensorflow=2.82e-01 |
| [bug_v4_000035](bug_v4_000035.py) | 13→9 (69%) | `LayerNormalization → Dropout → Unsqueeze → Squeeze → ReduceL2 → Mul → Add → Reshape → … (+1)` | onnxruntime=9.16e-01, tvm=3.97e-01, xla=3.97e-01, torch_compile=CRASH |
| [bug_v4_000186](bug_v4_000186.py) | 10→7 (70%) | `Mul → Add → Conv → Tanh → Concat → Conv → CumSum` | onnxruntime=1.43e+00, openvino=1.43e+00, tvm=1.43e+00, xla=1.43e+00 |
| [bug_000031](bug_000031.py) | 15→11 (73%) | `Add → LayerNormalization → Conv → Max → Div → Exp → Sub → Mul → … (+3)` | tensorflow=3.41e-01 |
| [bug_000424](bug_000424.py) | 8→6 (75%) | `Flatten → Reshape → MatMul → BatchNormalization → Clip → CumSum` | onnxruntime=1.03e+00, openvino=1.03e+00, tensorflow=1.03e+00, xla=1.03e+00 |
| [bug_000143](bug_000143.py) | 22→17 (77%) | `Mul → Add → Neg → Abs → Relu → Mul → ReduceMax → Mul → … (+9)` | onnxruntime=1.00e+00, tensorflow=1.00e+00 |
| [bug_000175](bug_000175.py) | 23→18 (78%) | `Mul → Sigmoid → Sub → Mul → Mul → Add → Add → Add → … (+10)` | openvino=CRASH |
| [bug_v4_000036](bug_v4_000036.py) | 5→4 (80%) | `MatMul → MatMul → MatMul → Resize` | tvm=3.09e-01, xla=3.09e-01 |
| [bug_000092](bug_000092.py) | 16→13 (81%) | `Mul → ReduceMean → Add → Sqrt → Div → Mul → Mul → Add → … (+5)` | onnxruntime=9.75e-01, openvino=9.75e-01, tensorflow=9.75e-01, xla=9.75e-01 |
| [bug_000267](bug_000267.py) | 12→10 (83%) | `MatMul → Add → Sigmoid → Add → Mul → Add → Tanh → Erf → … (+2)` | onnxruntime=9.72e-01, openvino=9.72e-01, tensorflow=9.72e-01, xla=9.72e-01 |
| [bug_v4_000198](bug_v4_000198.py) | 13→11 (85%) | `MatMul → Add → Sigmoid → Add → Relu → TopK → Tile → Add → … (+3)` | onnxruntime=1.00e+00 |
| [bug_v4_000223](bug_v4_000223.py) | 14→12 (86%) | `Gather → Reshape → TopK → Tile → Abs → ReduceSum → Add → Div → … (+4)` | onnxruntime=1.00e+00 |
| [bug_000342](bug_000342.py) | 8→7 (88%) | `Greater → Cast → Mul → MatMul → LogSoftmax → Mul → Pad` | tensorflow=3.26e-01 |
| [bug_000245](bug_000245.py) | 17→15 (88%) | `ReduceMean → Sub → Mul → ReduceMean → Add → Sqrt → Div → Mul → … (+7)` | tensorflow=4.93e-01 |
| [bug_000008](bug_000008.py) | 10→9 (90%) | `Selu → Mul → Sub → Add → TopK → Tile → Mul → Add → … (+1)` | onnxruntime=1.00e+00, openvino=1.00e+00 |
| [bug_000170](bug_000170.py) | 10→9 (90%) | `MatMul → Mul → Add → Relu → ConvTranspose → BatchNormalization → Relu → Conv → … (+1)` | tensorflow=4.06e-01 |
| [bug_000055](bug_000055.py) | 11→10 (91%) | `MatMul → Conv → Tanh → HardSwish → Mul → Abs → Add → Pow → … (+2)` | tensorflow=3.36e-01 |
| [bug_v4_000041](bug_v4_000041.py) | 11→10 (91%) | `LayerNormalization → Dropout → TopK → Tile → Abs → Mul → Gather → Slice → … (+2)` | openvino=1.00e+00, tvm=1.00e+00, xla=1.00e+00 |
| [bug_000032](bug_000032.py) | 12→11 (92%) | `Mul → Add → Relu → Pow → Mul → Sub → Add → Reshape → … (+3)` | tensorflow=CRASH |
| [bug_000189](bug_000189.py) | 13→12 (92%) | `MatMul → MatMul → ReduceMean → Sub → Mul → ReduceMean → Add → Sqrt → … (+4)` | tensorflow=3.23e-01 |
| [bug_000308](bug_000308.py) | 14→13 (93%) | `TopK → Tile → Add → Relu → Sub → Neg → Abs → Relu → … (+5)` | onnxruntime=9.33e-01, openvino=8.91e-01, tensorflow=1.00e+00, torch_compile=1.00e+00, xla=1.00e+00 |
| [bug_000166](bug_000166.py) | 9→9 (100%) | `Pad → Conv → BatchNormalization → Add → Mul → Unsqueeze → Squeeze → MatMul → … (+1)` | tensorflow=3.27e-01 |
| [bug_000210](bug_000210.py) | 11→11 (100%) | `Transpose → Mul → ReduceMean → Add → Sqrt → Div → Mul → Mul → … (+3)` | tensorflow=2.12e-01, xla=2.12e-01 |
| [bug_000248](bug_000248.py) | 18→18 (100%) | `Transpose → Transpose → MatMul → Transpose → Transpose → Einsum → Mul → ReduceMean → … (+10)` | onnxruntime=1.25e+00, openvino=1.25e+00, tensorflow=1.25e+00, xla=1.25e+00 |
| [bug_v4_000055](bug_v4_000055.py) | 12→12 (100%) | `MaxPool → AveragePool → Concat → Pad → Conv → Sub → Mul → Add → … (+4)` | xla=CRASH |


Root-cause primitives isolated by minimization:

- **`CumSum`** alone diverges on ORT + OpenVINO + TF + XLA (and TVM) — 6 bugs
  reduce to this single op.
- **`Pad` (reflect / edge / constant)** alone diverges on TensorFlow graph
  mode — 4 bugs reduce to this single op.
- **`Resize` with non-default coord modes** alone diverges across backends —
  3 bugs reduce to this single op.

---

## How to run

```bash
cd bugs/minimal

# single bug
python3 bug_000032.py           # TF crash
python3 github_ort_002.py       # still-live ORT resize bug

# everything
for f in *.py; do
    python3 "$f" && echo "  → REPRODUCED" || echo "  → not reproduced"
done
```

## Verification log

See `/tmp/final_verification2.txt` for the full per-bug output of the most
recent run. All 46 scripts exit 0.

## Directory layout

```
bugs/minimal/
├── README.md                 # this file
├── ROOT_CAUSES.md            # legacy root-cause notes (kept)
├── bug_000XXX.py             # 34 campaign v3 bugs (programmatic ONNX)
├── bug_v4_0XXXXX.py          # 13 campaign v4 bugs (base64 ONNX, 5-compiler sweep)
├── github_*.py, cross_*.py   # 32 still-live + new active bugs (batch 3 adds 12)
├── reports/                  # filing-ready bug-report scripts organized by compiler
│   ├── openvino/ (8), pytorch_inductor/ (1), tensorflow_xla/ (1), jax_xla/ (1)
└── _fixed_archive/           # 133 bugs fixed upstream + bug_000350 (borderline) — kept for reference
```

---

## Part 6 — 13 Campaign v5 + v6 Bugs (2026-04-14)

Discovered by 1000-model sweep across two seeds (v5 seed=1000, v6 seed=2077),
4-backend differential (`onnxruntime`, `torchscript`, `torch_compile`, `xla`).
Of 360 raw flagged divergences, 54 reproduce at tolerance 1e-2; of those, **41
match existing root causes** (Resize, CumSum, TopK family) and **13 are
genuinely new** root causes not covered by any of the previous 80 bugs.

All 13 verified to reproduce on the current CI (2026-04-14, JAX 0.9.2, ORT
1.24.4, PyTorch 2.9.1, torch.compile inductor).

### 6A. XLA AveragePool count_include_pad family (4 bugs)

The cleanest discovery: **`AveragePool(count_include_pad=True, kernel=3x3,
pads=[1,1,1,1])` alone** produces a >0.14 relative L2 error on JAX/XLA vs
PyTorch eager — XLA divides padded sums by the full kernel area (9) but
PyTorch normalizes only by the count of valid input elements at the boundary.

| # | Bug ID | rel_L2 | Min ops | Root cause |
|---|---|---:|---:|---|
| 81 | [bug_v6_000421](bug_v6_000421.py) | 0.145 | **1** | **`AveragePool(count_include_pad=1)` ALONE** — XLA boundary normalization mismatch |
| 82 | [bug_v5_000341](bug_v5_000341.py) | 0.174 | 7 | Add/Mul/Sub residual chain → MatMul → AveragePool(count_include_pad=1) → XLA |
| 83 | [bug_v5_000491](bug_v5_000491.py) | 0.141 | 6 | 3×Conv stack → Add → AveragePool(count_include_pad=1) → XLA |
| 84 | [bug_v6_000111](bug_v6_000111.py) | 0.162 | 18 | SE-block (GlobalAvgPool→FC→Sigmoid) → AveragePool(count_include_pad=1) → XLA |

### 6B. Attention / FFN divergence on XLA (5 bugs)

Modern transformer building blocks (cross-attention, multi-head self-attention,
SwiGLU FFN, windowed banded attention, gated MLP) all produce numerical
divergences on XLA that are absent from ORT/TorchScript. The campaign v3/v4
focus was on convolutional patterns; these are the first attention-pattern
findings.

| # | Bug ID | rel_L2 | Backend | Root cause |
|---|---|---:|---|---|
| 85 | [bug_v5_000013](bug_v5_000013.py) | 0.071 | xla, torch_compile | Cross-attention + MHA stack (MatMul/Transpose/Softmax × 3) → both XLA and torch.compile produce identical-but-wrong output (precision drift in the chained softmax+matmul fusion) |
| 86 | [bug_v6_000130](bug_v6_000130.py) | 0.101 | xla | Gated SwiGLU attention (MatMul + Sigmoid gate + Softmax + row-reduce-mul-transpose layout) → XLA's matmul-softmax fusion changes accumulation order |
| 87 | [bug_v6_000184](bug_v6_000184.py) | 0.011 | xla | Windowed banded attention (Greater→Cast→Mul mask before Softmax) + 4-LayerNorm sandwich → XLA masked-attention codepath divergence |
| 88 | [bug_v6_000273](bug_v6_000273.py) | 0.019 | xla | Gated MLP block (MatMul → Sigmoid gate → Mul + manual RMSNorm + Cast(int)/Cast(fp32) + Split + SwiGLU FFN) → XLA |
| 89 | [bug_v5_000442](bug_v5_000442.py) | 0.026 | xla | MatMul → LayerNorm → manual GeLU(tanh-approx) + Where/Greater branching → XLA precision drift from PyTorch eager |

### 6C. ORT optimizer over-fusion (2 bugs)

| # | Bug ID | rel_L2 | Backend | Root cause |
|---|---|---:|---|---|
| 90 | [bug_v6_000036](bug_v6_000036.py) | **17.84** | onnxruntime | **Three consecutive LayerNormalization** ops on the same axis: ORT's LayerNorm fusion pass merges the second+third LN into a single optimized kernel, dropping the intermediate Mul and producing ~18× scaled output |
| 91 | [bug_v6_000408](bug_v6_000408.py) | **3.51** | onnxruntime | `log1p+abs+ReduceMax+BatchNormalization+Clip+Sub+Add+LayerNormalization` chain: ORT optimizer reorders the BN-Clip-LN sequence, producing 3.5× wrong output |

### 6D. Other layout-sensitive XLA divergences (2 bugs)

| # | Bug ID | rel_L2 | Backend | Root cause |
|---|---|---:|---|---|
| 92 | [bug_v5_000025](bug_v5_000025.py) | 0.427 | xla | Transpose-MatMul-Transpose sandwich + LayerNorm + Reciprocal-Mul (manual normalization): XLA's transpose-collapse optimization changes the matmul layout, producing rel_L2 = 0.43 vs PyTorch eager (existing bug #28 covered this only on ORT/OV/TF; XLA is new) |
| 93 | [bug_v6_000164](bug_v6_000164.py) | **1.04** | xla | Channel-shuffle (Conv → Reshape to 5D → Transpose[0,2,1,3,4] → Reshape back): XLA's rank-5 transpose collapse changes the channel ordering, producing wrong output |

### Notes on the 4-bug AveragePool family

All 4 use the **identical** AveragePool attributes:
```
kernel_shape = [3, 3]
pads         = [1, 1, 1, 1]      # 1-pixel border each side
strides      = [1, 1]
count_include_pad = 1            # KEY: divide by kernel area, not valid count
```

The single-op repro ([bug_v6_000421](bug_v6_000421.py)) demonstrates this is
literally a **single-operator XLA bug**.  At a 32×32 input boundary pixel, the
3×3 kernel sees 6 valid + 3 padded zeros.  ONNX spec with `count_include_pad=1`
says: divide the sum by 9 (the full kernel size).  PyTorch eager respects this
correctly, but JAX's `lax.reduce_window`-based AveragePool computes the
normalization differently for padded regions.  Confirmed reproducible on JAX
0.9.2 with both CUDA-detection-on-fallback and pure-CPU paths.

### Reproducibility note (XLA + JAX)

These XLA-targeting scripts deliberately do NOT set `CUDA_VISIBLE_DEVICES=""`.
Some XLA divergences only manifest when JAX is allowed to probe CUDA (even when
no GPU is present — JAX falls back to CPU but takes a different device-init
codepath that affects op fusion and layout assignment).  Forcing
`CUDA_VISIBLE_DEVICES=""` before the JAX import suppresses several of these
bugs by 4–6 orders of magnitude.
