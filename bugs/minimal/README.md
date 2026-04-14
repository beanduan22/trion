# Minimal Reproducible Bug Scripts — Real Bugs Only

**46 self-contained Python scripts** — every bug verified to reproduce on current
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
| **Total real bugs** | **46** | all reproduce on current CI |

Tested on: Python 3.13, ONNX 1.21, ORT 1.24.4 (CPU), OpenVINO 2026.0,
TensorFlow 2.21.0 (CPU), PyTorch 2.9.1 (torch.compile inductor, torch.jit),
JAX 0.9.2, onnx2torch.

Uniqueness: each of the 46 bugs has a distinct ONNX graph signature (op
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

## Part 2 — 34 Campaign v3 Bugs

All oracle-verified; verdict is `rel_L2(backend vs pytorch_eager) > 0.1` or a
crash in the targeted backend.

### 2A. Crashes (4)

| # | Bug ID | Compiler | Root cause |
|---|---|---|---|
| 13 | [bug_000032](bug_000032.py) | TensorFlow (opt + noopt) | SpaceToDepth reshape/transpose feeds Conv with asymmetric pads `[0,1,0,1]` → `conv2d` dtype mismatch |
| 14 | [bug_000175](bug_000175.py) | OpenVINO (opt + noopt) | Gated residual + Conv1x1 + Split + manual GEGLU (Div/Erf/Add/Mul) + Conv2x2(s=2)+BN+LeakyRelu → CPU plugin internal error |
| 15 | [bug_000227](bug_000227.py) | TorchScript+opt | MaxPool `dilations=[2,2]` after Resize(asymmetric)+Conv3x3 branch → freeze + optimize_for_inference crash |
| 16 | [bug_000310](bug_000310.py) | TorchScript+opt | MaxPool `dilations=[2,2]` after conv-BN-fusion-both-paths + identity Transpose `[0,1,2,3]` |

### 2B. Multi-backend numerical divergence — ORT + OpenVINO + TF + XLA (±torch) (11)

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 17 | [bug_000036](bug_000036.py) | 0.31 | Resize(cubic, half_pixel) → Expand → Mul → Resize(linear, asymmetric) → LayerNorm |
| 18 | [bug_000060](bug_000060.py) | 0.57 | CumSum(last_axis) + SiLU residual + gated residual + Gather/Slice/Expand |
| 19 | [bug_000092](bug_000092.py) | 0.97 | Manual cRMS-norm (Mul/ReduceMean/Sqrt/Div) + ReLU residual + CumSum + Gather |
| 20 | [bug_000121](bug_000121.py) | 0.50 | CumSum → MatMul → ReduceL2 → MatMul → MatMul (triple matmul) |
| 21 | [bug_000143](bug_000143.py) | 1.00 | Neg+Abs+Relu sign-manip → ReduceMax → gated residual → manual LayerNorm (ORT+TF) |
| 22 | [bug_000163](bug_000163.py) | 1.10 | GLU Split → LogSoftmax → Mul → CumSum → Ceil → Cast(fp32→int32→fp32) roundtrip |
| 23 | [bug_000216](bug_000216.py) | 0.68 | CumSum + matmul-scale-add + Pow(canonical = x·x) + RMSNorm |
| 24 | [bug_000248](bug_000248.py) | 1.25 | Transpose-MatMul-Transpose sandwich + Einsum `bij→bji` + power-norm + CumSum |
| 25 | [bug_000267](bug_000267.py) | 0.97 | matmul+bias+Sigmoid + add-mul-add + Tanh+Erf + CumSum + Pow(x,1) identity |
| 26 | [bug_000308](bug_000308.py) | 1.00 | TopK(k=1)+Tile → Add+Relu+Sub → Neg+Abs+Relu → InstanceNorm1D (5 backends incl. torch.compile) |
| 27 | [bug_000424](bug_000424.py) | 1.01 | Flatten(axis=2) + MatMul + BN + Clip(Relu6) + CumSum + Conv3x3 + Elu |

### 2C. TF-only divergence (15)

All show rel_L2 ≥ 0.1 on TF graph mode vs PyTorch eager.

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 28 | [bug_000030](bug_000030.py) | 0.33 | Pad(reflect) + InstanceNorm + LayerNorm(axis=-1) + MatMul + Pow |
| 29 | [bug_000031](bug_000031.py) | 0.22 | Add→LayerNorm→Conv + unrolled CELU (Max+Div+Exp) + reflect-Pad + Conv + MatMul |
| 30 | [bug_000055](bug_000055.py) | 0.23 | MatMul + Conv1x1 + Tanh + HardSwish + Abs/Add/Pow/Sqrt + reflect-Pad + Conv3x3 |
| 31 | [bug_000166](bug_000166.py) | 0.33 | constant-Pad + Conv3x3 + BN + Unsqueeze/Squeeze (rank-manip) + MatMul + reflect-Pad |
| 32 | [bug_000170](bug_000170.py) | 0.25 | MatMul+Mul+Add+Relu + ConvTranspose2x2(s=2) + BN + Relu + asymmetric Conv pads `[0,1,0,1]` |
| 33 | [bug_000189](bug_000189.py) | 0.20 | MatMul+MatMul + manual LayerNorm → native LayerNorm (double norm) + Relu + reflect-Pad + Conv3x3 |
| 34 | [bug_000223](bug_000223.py) | 0.20 | Expand broadcast + Add + reflect-Pad + Conv3x3 + MatMul + Conv5x5 + Conv+BN+Elu |
| 35 | [bug_000232](bug_000232.py) | 0.35 | MatMul + reflect-Pad + channel-shuffle (Reshape/Transpose[0,2,1,3,4]/Reshape) + MatMul + Concat + Conv1x1 |
| 36 | [bug_000242](bug_000242.py) | 0.19 | MatMul+MatMul + reflect-Pad + Conv3x3 + Conv1x1+BN+Clip (Relu6) + GLU Split |
| 37 | [bug_000245](bug_000245.py) | 0.49 | Manual LayerNorm + power-norm (Pow/Div/Mul) + MatMul + edge-Pad + Reshape-flatten |
| 38 | [bug_000307](bug_000307.py) | 0.33 | reflect-Pad + Reshape-flatten + Conv1x1(s=2) + BN + LeakyRelu + MatMul + Relu+Add+Relu |
| 39 | [bug_000322](bug_000322.py) | 0.23 | reflect-Pad + Conv3x3 + MatMul + Conv3x3 + Selu + manual mean-variance norm + Resize(nearest, round_prefer_floor) |
| 40 | [bug_000342](bug_000342.py) | 0.41 | Greater + Cast(bool→fp32) mask + MatMul + LogSoftmax + reflect-Pad + edge-Pad |
| 41 | [bug_000372](bug_000372.py) | 0.28 | Tanh+Erf+Mul + BN-Conv-BN sandwich + Reshape-squash + edge-Pad + mul-self-as-pow |
| 42 | [bug_000416](bug_000416.py) | 0.21 | Two Transpose[2,3,0,1] squash-to-identity + ReduceL2 + Div + Mul + reflect-Pad + Conv + Pow + ReduceMean(axis=0) |

### 2D. TF + XLA divergence (2)

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 43 | [bug_000210](bug_000210.py) | 0.21 | Transpose[0,1,3,2] + RMSNorm + MatMul + Resize(linear, align_corners) |
| 44 | [bug_000404](bug_000404.py) | 0.35 | Resize(linear, align_corners) + Reshape-roundtrip + Conv3x3+BN+Elu + MatMul + InstanceNorm |

### 2E. ORT + OpenVINO only (2)

| # | Bug ID | rel_L2 | Root cause |
|---|---|---:|---|
| 45 | [bug_000008](bug_000008.py) | 1.00 | Selu → TopK(k=1) → Tile → Mul → Add → LayerNorm + Dropout (one-hot masking chain) |
| 46 | [bug_000426](bug_000426.py) | 0.47 | Resize(linear, asymmetric) + ASPP-style dilated convs (d=1, 2, 4) + BN + LeakyRelu + Split + Mul + Concat |

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
├── github_*.py, cross_*.py   # 12 still-live legacy bugs
└── _fixed_archive/           # 133 bugs fixed upstream + bug_000350 (borderline) — kept for reference
```
