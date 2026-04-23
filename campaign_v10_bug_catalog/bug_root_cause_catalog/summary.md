# Root-Cause Catalog — Summary

**Source:** `campaign_v10_results/bugs_unique/` (validated, deduplicated)
**Unique bugs processed:** 110
**Env-only cases excluded:** 0 (validation pass already filtered the 46 environmental failures before dedup)

## Category counts

| Category | Count |
|----------|-------|
| ONNX conversion/export bug | 0 |
| Compiler/backend bug       | 110 |
| Crash bug                  | 0 |

**Why zero in two categories:**
- **ONNX conversion/export bugs** would show ORT disagreeing with pytorch_eager. In every one of the 110 bugs, ORT sits inside the reference cluster (agrees with pytorch_eager), so none qualify.
- **Crash bugs** — the dedup/validation stage required a numerical divergence > tolerance. Pure-crash cases would not produce a repro at all; 0 of the 110 validated repros fail with a crash as the primary symptom. 28 of them also trigger a secondary crash in an *optimized* build of one of their already-diverging backends; those are co-symptoms of the same root cause and are not counted as a separate crash class.

## Root causes (by count)

| RC-ID | Count | Confidence | Root cause |
|-------|-------|-----------|------------|
| RC-01 | 41 | high | xla hlo attention matmul transpose softmax fusion |
| RC-02 | 26 | high | xla and inductor conv bn relu fusion |
| RC-03 | 5 | medium | fp16 cast roundtrip and softmax stability fusion |
| RC-04 | 5 | medium | inductor conv relu softmax chain |
| RC-05 | 5 | medium | tf graphmode attention softmax broadcast |
| RC-06 | 5 | medium | onnx spec squeeze unsqueeze bias softmax fusion ambiguity |
| RC-07 | 4 | high | topk k1 plus layernorm axis shape handling |
| RC-08 | 3 | medium | resize nearest asymmetric or depthwise conv bn fold |
| RC-09 | 3 | medium | xla only resize pool squeeze lowering |
| RC-10 | 2 | high | resize nearest round prefer floor semantics |
| RC-11a | 1 | low | row reduce mul transpose softmax layernorm misfold |
| RC-11b | 1 | low | reciprocal zero and conv bn fuse corner |
| RC-11c | 1 | low | tf and inductor resize linear halfpixel |
| RC-11d | 1 | low | branch min max plus layernorm residual |
| RC-11e | 1 | low | foldable add zero matmul triple constant folding |
| RC-11f | 1 | low | residual add relu plus attention matmul 4d except ts |
| RC-11g | 1 | low | spatial attention cbam reduce max ambiguity |
| RC-11h | 1 | low | expand add mul with layernorm axis1 fold |
| RC-11i | 1 | low | expm1 bounded plus resize layout |
| RC-11j | 1 | low | reciprocal neg zero with cast chain |
| RC-11k | 1 | low | torchscript only reduce l2 manual layernorm |

## Major clusters

- **RC-01 (41)** — TF + XLA (shared XLA HLO backend) diverge on attention/softmax/matmul pipelines. Everyone else agrees.
- **RC-02 (26)** — TF + torch.compile + XLA (three aggressive optimizers) diverge on conv + BatchNorm + Relu fusion chains.
- **RC-06 (5)** — Only ORT + pytorch\_eager agree; every compiler backend gives a different answer. Spec-ambiguous Squeeze/Unsqueeze + bias-softmax fusions.
- **RC-07 (4)** — `TopK(k=1)` + LayerNorm chain; TVM + ORT preserve semantics, five compilers all fail with rel\_l2 = 1.0.

## Distribution of worst rel\_l2

| Range | Count |
|-------|-------|
| ≥ 1.0 (total mismatch) | 26 |
| ≥ 0.5 | 33 |
| ≥ 0.1 | 46 |
| < 0.1 | 64 |

## Dominant ONNX operators across all 110 bugs

Mul (103), Add (96), MatMul (80), Relu (52), Conv (49), Softmax (47), Sub (45), Transpose (42),
Div (41), LayerNormalization (37), ReduceMean (35), Concat (26), Sigmoid (26), Sqrt (24),
Reshape (23), Cast (23), Where (21), BatchNormalization (20), Slice (18), ReduceSum (16),
Resize (15), Greater (15), ReduceMax (13), Tanh (12), Exp (12), Unsqueeze (12).

## Files

- `summary.md` — this file.
- `root_cause_manifest.json` — per-bug structured assignment.
- `root_cause_summary.json` — per-root-cause aggregate.
- `catalog.md` — grouped-by-root-cause narrative.
- `repros/unique_XXXX_min.py` — minimal constructive repros, one per root cause (plus the original-source repro path recorded in the manifest for every unique bug).
- `excluded_cases.md` — environmental/non-semantic cases (empty; already filtered upstream).
