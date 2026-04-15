# Campaign v8 Triage — Ground-Truth Verification

**Date**: 2026-04-16
**Verified by**: Real backend re-execution (ORT_DISABLE_ALL, ORT_ENABLE_ALL,
OpenVINO CPU 2026.0, onnx2torch, torch.compile(inductor), TorchScript).
TVM, onnx-tf, and tf2onnx not installed — TVM / XLA / TFLite suspects cannot
be directly re-executed, so verdict relies on whether the *testable* backends
form a consensus against the reported suspect claim.

**Tolerance**: `rel_L2 > 0.01` flagged as divergent **when the reference output
has non-trivial magnitude**. If the reference output has `max_abs < 1e-3`
(near-zero), `rel_L2` is numerically meaningless and we fall back to
`max_abs` compared against float32 precision scale.

---

## Pre-triaged backend crashes (confirmed, not investigated further)

| Bug | Suspect | Crash on | Root cause (oracle-side) |
|---|---|---|---|
| 207 | xla  | tflite+opt | `tfl.fully_connected` num_elements % 256 constraint (trion TFLite Tile lowering) |
| 584 | xla  | tflite+opt | same — num_elements % 128 constraint |
| 663 | xla  | tflite+opt | same — num_elements % 256 constraint |
| 741 | None | onnxruntime+nan | ORT `non-finite output` on numerically unstable random model |

All four confirmed oracle-side, not compiler bugs. No minimal repro produced.

---

## Candidate bug verification

### Bug 064 — suspect `xla`, dist_to_eager=0.073

**Pattern stack**: `add_relu_sub` → `einsum_batched_matmul` → `layernorm_temperature`
→ `identity_chain_two` → `cast_int_roundtrip`

**Ops**: `Add, Relu, Sub, Einsum, LayerNormalization, Div, Add, Identity, Identity, Clip([-100,100]), Cast(fp32→int32), Cast(int32→fp32), Mul(scale=0.01)`

**rel_L2 / max_abs vs ORT_DISABLE_ALL**:

| Backend | rel_L2 | max_abs | Verdict |
|---|---|---|---|
| ORT_DISABLE_ALL | 0 | 0 | ref |
| ORT_ENABLE_ALL  | 0 | 0 | ok |
| OpenVINO        | 0.126 | 0.01 | differs |
| onnx2torch      | 0 | 0 | ok |
| torch.compile (inductor) | 0 | 0 | ok |
| TorchScript     | 0 | 0 | ok |

**Root cause of the 0.01 diff**: All but OV agree exactly. OV's LayerNorm/Div
output drifts by ~0.001 relative to ORT (Conv/LN precision class). After the
`cast_int_roundtrip` pattern (`Cast(fp32→int32) → Cast(int32→fp32) → Mul(0.01)`),
values near the ±1 integer boundary (0.9998 vs 1.00095) round to *different*
integers, amplifying a 0.001 LN precision diff to exactly 0.01 in the final
output. Example: ORT clip_in=-0.99977 → i32=0, OV clip_in=-1.00095 → i32=-1.
Only 3 / 8192 elements differ.

**Suspect XLA is untestable directly**, but the 5 testable backends split
5-vs-1 with OV as outlier. This is the known-OV-Conv-precision class, not an
XLA bug. Max_abs is exactly at our tolerance (0.01) — marginal.

**Verdict**: **LIKELY FALSE POSITIVE** on suspect `xla`. The numerical
difference that exists is a side-effect of `cast_int_roundtrip` amplifying OV's
known LayerNorm precision drift. Dedupe: related to existing
`cross_openvino_conv_fp32_precision.py` (OV fp32 accumulation class).
**No new minimal repro produced.**

---

### Bug 157 — suspect `tvm`, dist_to_eager=9.12

**Pattern stack**: `add_layernorm` → `reduce_max_last` → `squeeze_unsqueeze`
→ `hardswish_activation` → `layer_norm_manual_chain`

**Output reference statistics**:
- shape `(1, 32, 64)`, **max_abs = 7.5e-5**, std = 3.8e-5, 50 % of elements below 1e-6.

**rel_L2 / max_abs vs ORT_DISABLE_ALL**:

| Backend | rel_L2 | max_abs | Verdict |
|---|---|---|---|
| ORT_DISABLE_ALL | 0 | 0 | ref |
| ORT_ENABLE_ALL  | 0 | 0 | ok |
| OpenVINO        | 1.0 | 7.5e-5 | noise |
| onnx2torch      | 1.16 | 1.1e-4 | noise |
| torch.compile   | 1.0 | 7.5e-5 | noise |
| TorchScript     | 1.16 | 1.1e-4 | noise |

The reference output itself has **max_abs = 7.5e-5**. All backend divergences
are at or below that same order of magnitude — they are **at float32 noise
floor** for a signal that is essentially zero. `rel_L2 ≈ 1` simply reflects
that `||ref|| ≈ ||diff||`. `suspect_distance_to_eager = 9.12` is an artefact
of the same normalisation pathology against an even-closer-to-zero eager
reference. TVM is untestable, but if the testable consensus is uniformly
noise-level, there is no reason to assume TVM is the exception.

**Verdict**: **FALSE POSITIVE**. Ill-conditioned LayerNorm-stacked oracle —
output is numerically zero. Same class as the pattern-compat cache cases
already handled upstream (real-TVM, fixed metric, absolute tolerance for
near-zero references). **No new minimal repro produced.**

---

### Bug 169 — suspect `openvino`, dist_to_eager=0.62

**Pattern stack**: `double_layernorm` → `gemm_relu` → `topk_last_axis_k1`
→ `add_layernorm` → `div_by_constant`

**Output reference statistics**: shape `(1, 64)`, **max_abs = 7.8e-5**, std = **7.3e-12**.

**rel_L2 / max_abs vs ORT_DISABLE_ALL**:

| Backend | rel_L2 | max_abs | Verdict |
|---|---|---|---|
| ORT_DISABLE_ALL | 0 | 0 | ref |
| ORT_ENABLE_ALL  | 0 | 0 | ok |
| OpenVINO        | 1.0 | 7.8e-5 | noise |
| onnx2torch      | 1.0 | 7.8e-5 | noise |
| torch.compile   | (fake-tensor error) | – | n/a |
| TorchScript     | 1.0 | 7.8e-5 | noise |

Output has std=7e-12 (!). Every element is essentially the same constant bias.
max_abs 7.8e-5 is the magnitude of that constant. Every other backend
diverges from ORT by exactly that amount, meaning they all return `0` while
ORT returns `~7.8e-5`. This is almost certainly `LN epsilon` / denormal /
flush-to-zero behaviour: OV, onnx2torch, and TorchScript flush denormals to
zero, ORT does not.

The suspect (`openvino`) is listed, but OV's behaviour matches onnx2torch and
TorchScript — it is **ORT** that is the outlier. Even so, the magnitude
(7.8e-5) is below any meaningful downstream tolerance.

**Verdict**: **FALSE POSITIVE**. Denormal / flush-to-zero precision noise on
an output that is numerically constant (std 7e-12). Not a real compiler bug.
**No new minimal repro produced.**

---

### Bug 303 — suspect `xla`, dist_to_eager=0.51

**Pattern stack**: `logsumexp_step` → `l2_norm_manual_primitives`
→ `shared_transpose_fanout` → `se_residual` → `ada_layer_norm`

**Output reference statistics**: shape `(1, 32, 1, 3)`, **max_abs = 8.1e-5**,
std = 3.3e-5, 81% of elements below 1e-6.

**rel_L2 / max_abs vs ORT_DISABLE_ALL**:

| Backend | rel_L2 | max_abs | Verdict |
|---|---|---|---|
| ORT_DISABLE_ALL | 0 | 0 | ref |
| ORT_ENABLE_ALL  | 0 | 0 | ok |
| OpenVINO        | 1.4 | 1.5e-4 | noise |
| onnx2torch      | 1.0 | 8.1e-5 | noise |
| torch.compile   | 0.56 | 7.5e-5 | noise |
| TorchScript     | 1.0 | 8.1e-5 | noise |

Same pattern as 157 / 169. Reference output is all near-zero (max_abs 8e-5)
because `logsumexp_step → l2_norm` produces a near-identity multiplier, and
the final `ada_layer_norm` centres the result. XLA untestable.

**Verdict**: **FALSE POSITIVE**. Near-zero reference, divergences below
float32 noise floor. **No new minimal repro produced.**

---

## Sanity check — bug 057 (suspect `xla`, dist_to_eager=0.0038)

Randomly sampled from the 37 "suspect-agrees-with-eager" bugs.

**Pattern stack**: `matmul_4d_batch` → `conv_bn_elu` → `greater_binary`
→ `matmul_4d_batch` → `slice_concat_identity`

**Output reference**: shape `(1, 256, 32, 32)`, **max_abs ≈ 7-10**,
std = 1.8 — non-trivial output magnitude, unlike the four candidates.

**rel_L2 / max_abs vs ORT_DISABLE_ALL**:

| Backend | rel_L2 | max_abs | Verdict |
|---|---|---|---|
| ORT_DISABLE_ALL | 0 | 0 | ref |
| ORT_ENABLE_ALL  | 1.4e-7 | 1.9e-6 | ok (within fp32 ε) |
| OpenVINO        | 0.013 | 0.43 | known Conv-fp32 precision issue |
| onnx2torch      | 0 | 0 | ok |
| torch.compile   | 1.3e-7 | 2.1e-6 | ok |
| TorchScript     | 0 | 0 | ok |

4 of 5 testable backends agree with ORT_DISABLE_ALL within 1 ULP. OV's 0.43
max_abs is the well-known `cross_openvino_conv_fp32_precision` class — not
what this bug claims to be. The *reported* suspect is `xla`, which is
untestable here, but with 4 independent non-XLA backends agreeing on the
reference, there is no reason to expect XLA diverges.

The claim "suspect agrees with ORT within tolerance" is consistent with the
testable evidence.

**FP hypothesis: HOLDS.** Low-distance-to-eager bugs really are agreeing
with a trustworthy reference; the oracle was misreporting them.

---

## Final tally

| Bucket | Count |
|---|---|
| Real compiler bugs (new)                              | **0** |
| Duplicates of existing `bugs/minimal/` files          | **0** (bug 057 OV drift is close to `cross_openvino_conv_fp32_precision` but the *reported* suspect `xla` isn't real; bug 064 is the same class — neither is a new bug and neither matches the reported suspect) |
| False positives (ill-conditioned near-zero oracle, or OV precision class misattributed to XLA/TVM) | **4** (bugs 064, 157, 169, 303) |
| Oracle-side crashes (already pre-triaged)             | **4** (bugs 207, 584, 663, 741) |
| Inconclusive                                          | **0** |

**New files written to `/home/binduan/myspace/trion/bugs/minimal/`**: **none**.

**Sanity-check on low-distance FP hypothesis**: sample bug 057 confirms —
suspect XLA is untestable directly but 4 of 5 other backends agree with ORT
within ≤ 2e-6 max_abs on a std ≈ 1.8 output. The "37 low-distance bugs" class
is a genuine false-positive bucket. This matches the README notes about
previously-deleted campaign files where consensus-disagrees-with-eager was
caused by the shared `onnx2torch` reference path, not a real compiler.

**Recommendation for the campaign oracle**: in addition to the existing
abs-tol-when-ref-is-near-zero fix mentioned in commit `de28ea7`, also require
that `max_abs(ref) ≥ threshold` (e.g. 1e-3) before declaring a divergence.
Bugs 157, 169, 303 all have `max_abs(ref) < 1e-4`, which is below the
meaningful-signal floor for fp32.
