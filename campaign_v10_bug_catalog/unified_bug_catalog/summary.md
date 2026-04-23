# Unified Bug Catalog — Summary

Merge of:
- `bug_root_cause_catalog/` — 110 numerical-divergence bugs, 21 RC clusters
- `crash_root_cause_catalog/` — 5 reproducible crashes, 1 RC cluster

**All entries are real, reproduced bugs.** Environmental noise, flaky failures, corrupted artifacts, and anything that was excluded upstream are **not** here — see `excluded_from_unified.md`.

## Totals

| | Count |
|-|------:|
| Bug/case entries | **115** |
| Root-cause clusters | **22** |
| Representative minimal repros | **22** (one per cluster) |
| Average repro size | ~60 lines, ~2.5 KB |

## By category

| Category | Root-cause clusters | Bug cases |
|----------|-----:|-----:|
| numerical divergence | 21 | 110 |
| compiler crash       | 1 | 5 |
| **Total**            | **22** | **115** |

## By nature

| Nature | Bug cases |
|--------|----------:|
| backend_bug      | 90 |
| spec_ambiguity   | 10 |
| uncertain        | 10 |
| crash_bug        | 5 |

## By confidence

| Confidence | Bug cases |
|------------|----------:|
| high   | 78 |
| medium | 26 |
| low    | 11 |

## By affected backend/project (a bug may affect multiple)

| Project | # bug cases hitting it |
|---------|----------------------:|
| xla | 95 |
| tensorflow | 92 |
| torch_compile | 54 |
| torchscript | 21 |
| openvino | 21 |
| tvm | 20 |

## Source catalog

| Catalog | Cases | RC clusters |
|---------|-----:|-----:|
| `bug_root_cause_catalog`   | 110 | 21 |
| `crash_root_cause_catalog` |   5 |  1 |

## Repro standard

Every minimal repro in `repros/` is:

- a single Python file, <120 lines, <5 KB;
- builds its graph programmatically with tiny shapes (no ONNX blobs, no base64, no recorded giant models);
- either constructs the graph via `onnx.helper` for divergence bugs or via `relay` ops for CR-01;
- passes `onnx.checker.check_model` and/or runs on ONNX Runtime or TVM in the repo's existing envs;
- for divergence bugs, calls the reference backend (ORT / pytorch_eager) and the suspect backend(s) via the existing `xcomp.oracle.*` backend classes and prints pairwise rel_l2;
- for CR-01, exits 0 when the free-variable error is reproduced.

## Files

- `summary.md` — this file.
- `catalog.md` — one compact section per root-cause cluster, grouped by nature.
- `manifest.json` — per-bug entries (115 total).
- `root_cause_summary.json` — per-RC entries (22 total).
- `repros/rcNN_min.py`, `repros/cr01_min.py` — minimal constructive repros, one per RC.
- `repros/_shared.py` — common helpers for the divergence repros.
- `excluded_from_unified.md` — what was left out and why.
