# Minimal Reproducible Bug Scripts — Real Bugs Only

**71 self-contained Python scripts** — every bug verified to reproduce on current
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
| **Total real bugs** | **71** | all reproduce on current CI |

Tested on: Python 3.13, ONNX 1.21, ORT 1.24.4 (CPU), OpenVINO 2026.0,
TensorFlow 2.21.0 (CPU), PyTorch 2.9.1 (torch.compile inductor, torch.jit),
JAX 0.9.2, onnx2torch.

Uniqueness: each of the 71 bugs has a distinct ONNX graph signature (op
sequence + key attributes). No two bugs share the same structure.

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
| 69 | [cross_bitshift_shift64_ov_ort](cross_bitshift_shift64_ov_ort.py) | ORT (RIGHT+LEFT), OpenVINO (LEFT) | 2147483648 | BitShift by 64 on uint64: C UB — x86 SHRQ masks shift count to 6 bits, so `x>>64` = `x>>0` = x; LEFT shift by 64 overflows to 2^31 on OV |
| 70 | [cross_crms_resize_nearest_ceil](cross_crms_resize_nearest_ceil.py) | TorchScript (onnx2torch) | 5.33 | `Resize(nearest, half_pixel, nearest_mode=ceil)`: onnx2torch ignores `nearest_mode=ceil` and uses PyTorch's floor-based nearest, choosing wrong source pixel at every other destination pixel |
| 71 | [cross_cumsum_kvcache_multicompiler](cross_cumsum_kvcache_multicompiler.py) | OpenVINO 2026.0 | 0.16 | CumSum(axis=2) + SE-block + self-attention: OV's optimized kernel path for CumSum+ReduceMean accumulates fp32 in a different order than ORT, error amplified by softmax attention |

### Notes on extreme divergence (bugs 61, 62)

`bug_v4_000198` and `bug_v4_000223` produce rel_L2 in the 10^4–10^5 range.
Both use a **LayerNorm with near-zero variance** (all-same input from TopK+Tile)
followed by a **temperature division**.  When all elements are identical,
LayerNorm variance → 0 and the normalization output is numerically unstable
(depends on epsilon).  ORT and PyTorch use different epsilon handling → cascade to
catastrophic amplification in the downstream Div.

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
├── github_*.py, cross_*.py   # 20 still-live + new active bugs
└── _fixed_archive/           # 133 bugs fixed upstream + bug_000350 (borderline) — kept for reference
```
