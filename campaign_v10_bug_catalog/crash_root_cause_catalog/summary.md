# Crash-Root-Cause Catalog — Summary

**Source:** `campaign_v10_results/compiler_crashes/` (1,500-model campaign_v10).

## Totals

| Metric | Count |
|--------|------:|
| Crash **reports** on disk (`crash_*_report.json`) | 664 |
| ONNX models on disk (`crash_*.onnx`) | 664 |
| Crash **events** (one per backend × model) | 1,236 |
| Events **reproducible as real compiler bugs** | **5** |
| Events classified **environmental** (GPU OOM and downstream) | **1,190** |
| Events classified **flaky / non-reproducible** | **41** |
| Events classified **exporter/conversion crash** | 0 |
| Events classified **corrupted artifact / invalid input** | 0 |

## Classification breakdown

| Classification | # Events | Description |
|----------------|---------:|-------------|
| Reproducible backend/compiler crash | **5** | TVM Relay build aborts with "contains free variables" — 5/5 reproduced from the original ONNX *and* reproduced from a minimal 2-node repro in TVM 0.11.1. |
| Reproducible exporter/conversion crash | 0 | None found. |
| Environment / infrastructure failure | **1,190** | GPU CUDA OOM (`cudaSetDevice` out of memory, `CUDA_ERROR_OUT_OF_MEMORY`, device-side assert needing `TORCH_USE_CUDA_DSA`, in-memory CUBIN load failures). |
| Corrupted artifact / invalid input | 0 | None found inside this directory. (46 session-level env failures exist upstream — see `bug_root_cause_catalog/excluded_cases.md`.) |
| Non-reproducible / flaky / uncertain | **41** | 40 × XLA Triton autotuner "No valid config found" on small shapes (concurrent with the OOM storm — autotuner cache is GPU-state-polluted), 1 × torch.compile "cuDNN CUDNN_STATUS_INTERNAL_ERROR". Neither is cleanly reproducible on a fresh GPU. |

## Root-cause buckets

| RC-ID | # Events | Classification | Confidence | Root cause |
|-------|---------:|----------------|-----------|------------|
| **CR-01** | **5** | reproducible backend bug | **high** | TVM Relay build with `freeze_params=True` emits multiple `Var(name, …)` nodes for the same input when an intermediate tensor is used as both operands of a binary op — the free-var check then aborts. |
| CR-FLAKY-AUTOTUNE | 40 | non-reproducible / flaky | medium | XLA Triton autotuner reports `NOT_FOUND: No valid config found!` for `gemm_fusion_dot` / `cudnn-conv-bias-activation` on small shapes; strongly correlated with concurrent CUDA OOMs in the same session. |
| CR-FLAKY-CUDNN    | 1  | non-reproducible / flaky | low    | torch.compile raises `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`; co-occurs with another crash on the same model. GPU-state fallout. |
| CR-ENV-OOM        | 1,190 | environment               | high   | CUDA device out-of-memory and its downstream (device-side assertions, CUBIN load failures, constant-buffer allocation failures). |

## Event distribution across backends

| Backend | Crash events | Real bugs among them |
|---------|------:|------:|
| tensorflow (+opt) | 616 | 0 |
| torch_compile (+opt) | 609 | 0 |
| xla (+opt) | 6 | 0 |
| tvm (+opt) | 5 | **5** |

TF + torch.compile together account for 1,225/1,236 events — every one of them is OOM-driven (they are the GPU-running backends in this harness).

## Reproducibility (real bugs only)

| Model | Backend | Original repro | Minimal constructive repro |
|-------|---------|---------------|----------------------------|
| 385   | tvm+opt | ✅ | ✅ (2 ONNX nodes) |
| 905   | tvm+opt | ✅ | ✅ (same pattern) |
| 959   | tvm+opt | ✅ | ✅ (same pattern) |
| 1225  | tvm+opt | ✅ | ✅ (same pattern) |
| 1452  | tvm+opt | ✅ | ✅ (same pattern) |

All 5 original ONNX files reproduce the crash; all 5 also collapse to the **single 2-node minimal** `MatMul(x, W) → Add(m, m)` shown in `repros/cr01_tvm_free_vars_min.py`.

## Files

- `summary.md` — this file.
- `crash_manifest.json` — one entry per crash event (1,236 rows).
- `catalog.md` — grouped-by-root-cause narrative.
- `repros/cr01_tvm_free_vars_min.py` — minimal 2-node repro (self-contained, builds ONNX in code).
- `repros/cr01_tvm_free_vars_replay_original.py` — replays the 5 original ONNX models.
- `excluded_cases.md` — all environmental and flaky buckets, with exclusion reasons.
