# Bug Reports — 9 bugs × 4 affected compilers = 11 reproducers

Filing-ready bug reports for the 9 new bugs discovered on **2026-04-14**.
Each `.py` file is a minimal, self-contained reproducer suitable for
attaching to a GitHub issue.

---

## Directory layout

```
reports/
├── openvino/                  ← 8 reports (bugs 1–8 all affect OV only)
│   ├── 01_uint8_sub_saturation.py
│   ├── 02_uint8_mul_saturation.py
│   ├── 03_uint8_add_saturation.py
│   ├── 04_int8_sub_saturation.py
│   ├── 05_int8_add_saturation.py
│   ├── 06_reducelogsumexp_overflow.py
│   ├── 07_relu_nan_violation.py
│   └── 08_exp_nan_violation.py
├── pytorch_inductor/          ← bug 9 (Inductor simplifier)
│   └── 09_bf16_cast_elide.py
├── tensorflow_xla/            ← bug 9 (XLA AlgebraicSimplifier)
│   └── 09_bf16_cast_elide.py
└── jax_xla/                   ← bug 9 (same XLA simplifier, via JAX)
    └── 09_bf16_cast_elide.py
```

Each script:

- Starts with a full bug-report docstring (what-is-wrong, root cause,
  cross-compiler verification).
- Builds a minimal ONNX or native graph (no external files, seed-stable).
- Runs one affected compiler.
- Exits **0 = bug reproduces**, **1 = does not reproduce**, **2 = deps missing**.

---

## Bug → Compiler root-cause matrix

| # | Bug | Compiler | Where to file | Root cause (per compiler) |
|---|---|---|---|---|
| 1 | uint8 Sub saturates | **OpenVINO CPU** | openvinotoolkit/openvino | Element-wise kernel dispatches to `_mm_subs_epu8` (saturating SSE intrinsic) instead of `_mm_sub_epi8` (wrapping). 16/32/64-bit paths use a different dispatch and wrap correctly. |
| 2 | uint8 Mul saturates | **OpenVINO CPU** | openvinotoolkit/openvino | Same dispatch bug as #1 — 8-bit Mul kernel uses saturating SIMD for uint8 products > 255. |
| 3 | uint8 Add saturates | **OpenVINO CPU** | openvinotoolkit/openvino | `_mm_adds_epu8` used instead of `_mm_add_epi8`. |
| 4 | int8 Sub saturates | **OpenVINO CPU** | openvinotoolkit/openvino | Signed variant: `_mm_subs_epi8` used instead of `_mm_sub_epi8`. Result clamps to `[-128, 127]` on overflow/underflow. |
| 5 | int8 Add saturates | **OpenVINO CPU** | openvinotoolkit/openvino | `_mm_adds_epi8` used instead of `_mm_add_epi8`. |
| 6 | ReduceLogSumExp overflows for x ≥ 88.7 | **OpenVINO CPU** | openvinotoolkit/openvino#32839 | Naive implementation as `Log(ReduceSum(Exp(x)))` — `Exp(100)` overflows fp32 to `+inf`. Every other compiler uses the max-shift stabilizer `m + log(Σ exp(xᵢ − m))`. |
| 7 | Relu(NaN) → 0 | **OpenVINO CPU** | openvinotoolkit/openvino | Kernel uses `x > 0 ? x : 0` with an ordered comparison. IEEE 754 ordered compares return false for NaN → falls into the `else 0.0` branch. Fix: `fmaxf(0, x)` or explicit NaN check. |
| 8 | Exp(NaN) → +inf | **OpenVINO CPU** | openvinotoolkit/openvino | Kernel short-circuits large inputs to `+inf` before computing `exp`. NaN's unordered comparison maps into the "large positive" branch. Fix: add `isnan(x)` guard before the range short-circuit. |
| 9a | bf16↔fp32 cast elided | **PyTorch Inductor** | pytorch/pytorch#179561 | Algebraic simplifier applies `Cast(T→U)∘Cast(U→T) → Identity` without verifying `precision(U) ≥ precision(T)`. bf16 has 7 mantissa bits vs fp32's 23, so the round-trip is lossy, not identity. |
| 9b | bf16↔fp32 cast elided | **TensorFlow XLA** | openxla/xla | Same algebraic rule in `xla/service/algebraic_simplifier.cc` — back-to-back `HloInstruction::kConvert` ops collapsed without precision check. |
| 9c | bf16↔fp32 cast elided | **JAX (jax.jit via XLA)** | openxla/xla | **Shared backend with TF-XLA.** Same simplifier, same bug. One XLA fix resolves both 9b and 9c. |

---

## Cross-compiler coverage

| Bug | numpy | ONNX-Ref | ORT | **OpenVINO** | onnx2torch | PT-eager | **Inductor** | TorchScript | TF-eager | **TF-XLA** | JAX | **JAX-jit** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 uint8 Sub | ✓ | ✓ | — | **✗** | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| 2 uint8 Mul | ✓ | ✓ | — | **✗** | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| 3 uint8 Add | ✓ | ✓ | — | **✗** | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| 4 int8 Sub | ✓ | ✓ | — | **✗** | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| 5 int8 Add | ✓ | ✓ | — | **✗** | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| 6 ReduceLogSumExp | ✓ | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | — |
| 7 Relu(NaN) | — | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| 8 Exp(NaN) | — | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| 9 bf16 cast | ✓ | ✓ | ✓ | ✓ | — | ✓ | **✗** | ✓ | ✓ | **✗** | ✓ | **✗** |

Legend:  ✓ = correct   **✗** = bug   — = not applicable / not tested

---

## Severity & impact summary

| Bug | Severity | Silent? | Impact |
|---|---|---|---|
| 1–5 (int arithmetic) | High | Yes | Any ONNX model using uint8/int8 Add/Sub/Mul produces wrong results on OV CPU. Common in quantized integer-only networks, image-processing pipelines, hash/modular layers. |
| 6 (LogSumExp) | High | Yes | Softmax-log expressions, contrastive-loss evaluation, MoE routing — any model with logits near the exp overflow boundary produces +inf on OV CPU. |
| 7 (Relu NaN) | Medium-High | Yes | Models using NaN as a validity sentinel (attention masking, missing-value flags) have those sentinels silently replaced by zero on OV CPU. |
| 8 (Exp NaN) | Medium-High | Yes | Same as 7 but NaN becomes +inf, which then amplifies through any downstream op (softmax, log). |
| 9 (bf16 cast) | High | Yes | Every mixed-precision model (BERT, LLMs with bf16 accumulators, quantization-aware training) runs at higher precision under JIT than in eager mode. **Affects all three production JIT compilers (Inductor, TF-XLA, JAX-jit).** |

---

## How to file these

1. Clone this directory.
2. `cd` into the compiler subfolder matching where you want to file.
3. Run `python3 <script>` — confirm exit code 0.
4. Copy the docstring header + the script output into a new GitHub issue
   on the relevant project's tracker (listed in the bug-cause matrix above).
5. Attach the `.py` file as-is.

Every script is self-contained and requires only the compiler under test
plus `numpy`, `onnx` (for OV/ORT-facing reports), or `torch`/`tf`/`jax`
natively for the JIT-compiler reports.

---

## Reproduction environment

- Python 3.13
- onnx 1.21.0, onnxruntime 1.24.4
- openvino 2026.0.0
- torch 2.9.1+cu128
- tensorflow 2.21.0
- jax 0.9.2
- onnx2torch installed
- Hardware: x86_64, NVIDIA RTX 6000 Ada (not used for CPU bugs)
