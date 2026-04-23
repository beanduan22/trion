# Excluded from the Unified Catalog

The unified catalog contains only **real, reproducible bug cases**. The
items below were excluded by intent. Counts cite the upstream source
catalogs and match their exclusion notes.

## From `crash_root_cause_catalog/` (1,236 total events)

| Excluded bucket | # Events | Reason |
|------------------|---------:|--------|
| CUDA out-of-memory and downstream device-side assertions | 1,190 | Environmental — GPU memory pressure in the long-running campaign session, zero model signal. |
| XLA Triton autotuner "No valid config found" | 40 | Flaky — 8 of a 9-case fresh-GPU replay ran clean (0 reproduced). Strongly correlated with the concurrent OOM storm above. |
| torch.compile `CUDNN_STATUS_INTERNAL_ERROR` | 1 | Flaky — replayed clean on a fresh GPU; same model also tripped a separate flaky event in the campaign, indicating GPU-state fallout. |

**Total excluded crash events: 1,231 / 1,236.** Kept: **5 / 1,236** (all CR-01).

## From `bug_root_cause_catalog/` upstream pipeline

| Excluded bucket | # Cases | Reason |
|------------------|-------:|--------|
| Validation failures (dedup Pass 1+2) | 46 | Environmental before dedup ever ran: `CUDAExecutionProvider` not in available providers on the CPU host, corrupted-zip PyTorch-export artifacts, oneDNN diagnostic noise, stale-artifact revalidation misses. |
| Duplicates collapsed into the 110 unique buckets | 0 | Dedup produced 110 unique buckets from 110 genuine repros; no intra-bucket duplicates were discarded. |

## Explicitly kept, not excluded

`spec_ambiguity` and `uncertain` clusters are **kept** in the unified
catalog (they were already classified in the prior catalog's
`nature` field). The unified `manifest.json` carries the `nature` field
so downstream consumers can filter them if needed:

| Nature | # bug cases |
|--------|-----:|
| backend_bug      | 90 |
| spec_ambiguity   | 10 |
| uncertain        | 10 |
| crash_bug        |  5 |

## Tally

- **Unified catalog**: 115 cases / 22 root-cause clusters.
- **Excluded**: 1,231 crash events (env + flaky) + 46 upstream env failures.
- **Net**: every kept case has a working minimal repro and a root-cause
  assignment with a confidence level.
