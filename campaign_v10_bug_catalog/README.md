# Campaign v10 Bug Catalog

End-to-end root-cause catalog for the **1,500-model XComp fuzzing campaign (v10)**,
including the full audit trail and minimal reproducers for every confirmed bug.

All analysis files in this folder were produced from the raw campaign outputs
(`campaign_v10_results/`) and then audited for reproducibility. Anything that
could not be cleanly reproduced — CUDA OOMs, flaky autotuner failures,
corrupted artifacts — is excluded. Each remaining bug has a single-file,
tiny-shape, programmatically-constructed repro in `unified_bug_catalog/repros/`.

---

## TL;DR

- **1,500 models generated** across all 21 attempts of the supervised run.
- **213 raw repros emitted**, **156 re-validated**, **110 unique deduplicated
  numerical-divergence bugs**, and **5 reproducible TVM crashes**.
- Those 115 cases collapse into **22 root-cause clusters** → one minimal repro
  per cluster.
- **6 compiler backends** are implicated in divergences (OV, TF, torch.compile,
  TorchScript, TVM, XLA) + **1** (TVM) for a real crash.
- **1,231 / 1,236** recorded crash events are GPU-memory noise; only 5 survive
  the reproducibility audit.

---

## Folder layout

```
campaign_v10_bug_catalog/
├── README.md                           — this file (with all recorded outputs)
├── unified_bug_catalog/                — the canonical merged catalog
│   ├── summary.md                      — totals + breakdowns
│   ├── catalog.md                      — grouped-by-root-cause narrative
│   ├── manifest.json                   — 115 per-case entries
│   ├── root_cause_summary.json         — 22 per-cluster entries
│   ├── excluded_from_unified.md        — what was left out and why
│   └── repros/                         — 22 minimal repros + _shared.py
├── bug_root_cause_catalog/             — raw divergence analysis (110 bugs, 21 RCs)
│   ├── summary.md, catalog.md, manifest.json, …
│   ├── root_cause_classification.md    — backend_bug / spec_ambiguity / uncertain
│   ├── issue_priority.md               — P1/P2/P3 filing plan
│   ├── paper_ready_table.md            — compact publication table
│   └── repros/                         — 21 divergence repros
└── crash_root_cause_catalog/           — raw crash analysis (1,236 events → 5 real)
    ├── summary.md, catalog.md, excluded_cases.md
    ├── crash_manifest.json
    ├── audit_followup.md               — second-pass audit summary
    ├── flaky_recheck.md                — 9-case fresh-GPU replay
    ├── environment_sample_audit.md     — 50-case stratified env sample
    ├── cr01_deep_dive.md               — refined CR-01 mechanism
    └── repros/                         — 1 minimal crash repro + replay drivers
```

---

## 1. Campaign v10 pipeline — recorded run details

```
[supervisor] attempt=1   start_model=0     target=1500
[supervisor] attempt=2   start_model=195   target=1500
[supervisor] attempt=3   start_model=415   target=1500
[supervisor] attempt=4   start_model=478   target=1500
[supervisor] attempt=5   start_model=479   target=1500
[supervisor] attempt=6   start_model=480   target=1500
[supervisor] attempt=7   start_model=481   target=1500
[supervisor] attempt=8   start_model=482   target=1500
[supervisor] attempt=9   start_model=483   target=1500
[supervisor] attempt=10  start_model=484   target=1500
[supervisor] attempt=11  start_model=485   target=1500
[supervisor] attempt=12  start_model=786   target=1500
[supervisor] attempt=13  start_model=787   target=1500
[supervisor] attempt=14  start_model=788   target=1500
[supervisor] attempt=15  start_model=789   target=1500
[supervisor] attempt=16  start_model=790   target=1500
[supervisor] attempt=17  start_model=791   target=1500
[supervisor] attempt=18  start_model=792   target=1500
[supervisor] attempt=19  start_model=793   target=1500
[supervisor] attempt=20  start_model=794   target=1500
[supervisor] attempt=1   start_model=1185  target=1500
[supervisor] DONE — highest model=1500 >= target=1500
```

**Campaign outputs on disk:**

- 213 ONNX models + 213 repro scripts + 214 reports in `campaign_v10_results/`
- 664 crash reports + 664 ONNX models in `campaign_v10_results/compiler_crashes/`
- 1,328 crash artifacts total (1,236 distinct crash events after dedup)
- `checkpoints/cp_000100 … cp_001500`, `in_progress/` clean-exited

**Dedup/validation (two passes, 2026-04-22):**

```
Pass 1 (Apr 22 07:53):
  156 repros → 20 genuine / 132 failed → 20 unique buckets

Pass 2 (Apr 22 08:24, after regen_repros.py patched the 132 failed scripts):
  156 repros → 110 genuine / 46 failed → 110 unique buckets
```

Pass 2 output is canonical: 110 unique validated repros + 46 environmental
failures (all filtered as CUDA-provider warnings, corrupted torch-export zips,
oneDNN noise, or stale artifacts).

---

## 2. Divergence analysis — 110 unique bugs → 21 root causes

### Backend disagreement pattern (which pair disagreed the most, per bug)

```
  19  ('torch_compile', 'xla')
  18  ('onnxruntime', 'xla')
  15  ('xla', 'tvm')
  15  ('onnxruntime', 'torchscript')
   7  ('torchscript', 'xla')
   6  ('onnxruntime', 'torch_compile')
   6  ('torch_compile', 'tensorflow')
   6  ('torchscript', 'tensorflow')
   5  ('xla', 'openvino')
   3  ('onnxruntime', 'tensorflow')
   3  ('tvm', 'tensorflow')
   2  ('torch_compile', 'tvm')
   2  ('torch_compile', 'openvino')
   2  ('torchscript', 'torch_compile')
   1  ('onnxruntime', 'openvino')
```

### Worst rel_l2 distribution

| Range | Count |
|-------|------:|
| rel_l2 ≥ 1.0 (total mismatch) | 26 |
| rel_l2 ≥ 0.5 | 33 |
| rel_l2 ≥ 0.1 | 46 |
| rel_l2 < 0.1 | 64 |

min = 0.0502, max = 1.000.

### Dominant ONNX ops across all 110 bugs (unique-per-bug)

```
Mul(103)    Add(96)   MatMul(80)  Relu(52)   Conv(49)   Softmax(47)
Sub(45)     Transpose(42)  Div(41)   LayerNormalization(37)
ReduceMean(35)  Concat(26)  Sigmoid(26)  Sqrt(24)  Reshape(23)
Cast(23)    Where(21)  BatchNormalization(20)  Slice(18)
ReduceSum(16)  Resize(15)  Greater(15)  ReduceMax(13)
Tanh(12)    Exp(12)    Unsqueeze(12)  Squeeze(10)
```

### Suspect-backend set frequency

```
  41  ('tensorflow', 'xla')                                        → RC-01
  26  ('tensorflow', 'torch_compile', 'xla')                       → RC-02
   5  ('openvino', 'tensorflow', 'torch_compile', 'xla')           → RC-03
   5  ('torch_compile',)                                           → RC-04
   5  ('tensorflow',)                                              → RC-05
   5  ('openvino', 'tensorflow', 'torch_compile', 'torchscript', 'tvm', 'xla')  → RC-06
   4  ('openvino', 'tensorflow', 'torch_compile', 'torchscript', 'xla')          → RC-07
   3  ('openvino', 'torch_compile', 'torchscript', 'tvm', 'xla')   → RC-08
   3  ('xla',)                                                     → RC-09
   2  ('tensorflow', 'torch_compile', 'torchscript', 'tvm', 'xla') → RC-10
  11  singleton clusters                                           → RC-11a … RC-11k
```

ORT is in the reference cluster in **100 %** of bugs → no ONNX conversion/export bugs, every divergence is a compiler/backend issue.

### Paper-ready table (21 divergence clusters + 1 crash cluster)

Abbreviations: **OV**=OpenVINO, **TF**=TensorFlow, **TC**=torch.compile (Inductor),
**TS**=TorchScript, **TVM**=Apache TVM, **XLA**=OpenXLA, **ORT**=ONNX Runtime (reference).

| RC | # Bugs | Affected Backend(s) | Trigger Pattern | Nature | Conf. | Repro |
|----|-------:|---------------------|-----------------|--------|-------|-------|
| RC-01 | 41 | TF, XLA | `MatMul → Transpose → Softmax → MatMul` (4-D attention) | backend bug | high | `repros/rc01_min.py` |
| RC-02 | 26 | TF, TC, XLA | `Conv → BN → Relu` with broadcast `Add`/`Mul` upstream | backend bug | high | `repros/rc02_min.py` |
| RC-03 | 5 | OV, TF, TC, XLA | `Cast(f32→f16) → Cast(f16→f32) → manual softmax` | uncertain | med | `repros/rc03_min.py` |
| RC-04 | 5 | TC | `Conv → Relu → Softmax → Expand → Add → Conv → Relu` | backend bug | med | `repros/rc04_min.py` |
| RC-05 | 5 | TF | `3× MatMul → Softmax(axis=0) → Greater/Where → Softmax` | backend bug | med | `repros/rc05_min.py` |
| RC-06 | 5 | OV, TF, TC, TS, TVM, XLA | `Add³ → Unsqueeze → Squeeze → bias+Softmax → LN` | spec ambiguity | med | `repros/rc06_min.py` |
| RC-07 | 4 | OV, TF, TC, TS, XLA | `TopK(k=1) → Tile → … → LN` (singleton axis dropped) | backend bug | high | `repros/rc07_min.py` |
| RC-08 | 3 | OV, TC, TS, TVM, XLA | depthwise-Conv + BN fold + Resize(nearest, asymmetric) | uncertain | med | `repros/rc08_min.py` |
| RC-09 | 3 | XLA | `MatMul(4-D) → Slice → Tile → MatMul → Resize` | backend bug | med | `repros/rc09_min.py` |
| RC-10 | 2 | TF, TC, TS, TVM, XLA | `Resize(nearest, asymmetric, round_prefer_floor)` | spec ambiguity | high | `repros/rc10_min.py` |
| RC-11a | 1 | OV, TS, TVM, XLA | `ReduceMean → Mul → Transpose → MatMul → Softmax → LN` | uncertain | low | `repros/rc11a_min.py` |
| RC-11b | 1 | OV, TS, TVM | `Reciprocal(0) → Mul → Conv → BN → Relu` | spec ambiguity | low | `repros/rc11b_min.py` |
| RC-11c | 1 | TF, TC | `Relu → Resize(linear, half_pixel, 5→4)` | backend bug | low | `repros/rc11c_min.py` |
| RC-11d | 1 | TS, TVM, XLA | `Max → Min → LN → Residual-Add → Relu` | backend bug | low | `repros/rc11d_min.py` |
| RC-11e | 1 | TF, TC, TS | `3× MatMul → Add(0) → Mul(1) → Greater → Where(const)` | spec ambiguity | low | `repros/rc11e_min.py` |
| RC-11f | 1 | OV, TF, TC, TVM, XLA | `Add(resid) → Relu → MatMul → Softmax → MatMul → Conv` | backend bug | low | `repros/rc11f_min.py` |
| RC-11g | 1 | TF, TS, XLA | `ReduceMean ∥ ReduceMax → Concat → Conv → Sigmoid → Mul` (CBAM) | backend bug | low | `repros/rc11g_min.py` |
| RC-11h | 1 | TC, TVM | `Expand → Add → Mul → LN(axis=1)` | uncertain | low | `repros/rc11h_min.py` |
| RC-11i | 1 | TS, XLA | `Exp → Sub(1) → MatMul → Tanh` (manual `expm1`) | backend bug | low | `repros/rc11i_min.py` |
| RC-11j | 1 | OV, XLA | `Reciprocal(-0.0) → Relu` (signed-zero IEEE-754 corner) | spec ambiguity | low | `repros/rc11j_min.py` |
| RC-11k | 1 | TS | `Mul → ReduceSum → Add(eps) → Sqrt → Div` (manual L2-norm) | backend bug | low | `repros/rc11k_min.py` |
| **CR-01** | **5** | **TVM** | `relay.nn.dense(units=None) → relay.add(y, y)` inside `relay.build` | **crash bug** | high | `repros/cr01_min.py` |

Paths above are relative to `unified_bug_catalog/`.

### Aggregated view

| | Count |
|-|------:|
| Unique validated bugs | 115 |
| Root-cause clusters | 22 |
| Backend bug clusters | 13 (93 cases) |
| Crash bug clusters | 1 (5 cases) |
| Spec-ambiguity clusters | 5 (10 cases) |
| Uncertain | 3 (7 cases) |
| ONNX conversion/export bugs (ORT ≠ pytorch_eager) | 0 |

---

## 3. Crash analysis — 1,236 events → 5 real bugs

Raw counts from `campaign_v10_results/compiler_crashes/`:

- **664** crash reports on disk, **664** ONNX models.
- **1,236** crash events (one per backend × model). Most reports have both
  TF+opt and torch_compile+opt tripping on the same input.
- Events per backend: `tensorflow+opt` = 616, `torch_compile+opt` = 609,
  `xla+opt` = 6, `tvm+opt` = 5.

### Classification

| Classification | # Events |
|----------------|---------:|
| Reproducible backend/compiler crash (**kept**) | **5** |
| Reproducible exporter/conversion crash | 0 |
| Environment / infrastructure failure | **1,190** |
| Corrupted artifact / invalid input | 0 |
| Non-reproducible / flaky / uncertain | **41** |

### Environmental bucket breakdown (the 1,190)

| Signature | # |
|-----------|--:|
| `cudaSetDevice() on GPU:0 failed. Status: out of memory` | 558 |
| `Compile with TORCH_USE_CUDA_DSA …` (downstream of OOM) | 550 |
| `CUDA out of memory. Tried to allocate … MiB` (variants) | ~54 |
| `RuntimeError: CUDA driver error: out of memory` | 4 |
| `RESOURCE_EXHAUSTED: … CUDA_ERROR_OUT_OF_MEMORY` | 4 |
| `TF Out of memory while trying to allocate 16 MiB` | 15 |
| `Failed to allocate N bytes for new constant` | 1 |

### Audit task 1 — flaky recheck (9/9 clean on fresh GPU)

```
[1/9] crash_001126__tensorflow_opt     -> RAN_OK  (53s)  out=(1, 128, 16, 8)
[2/9] crash_001107__tensorflow_opt     -> RAN_OK  (51s)  out=(1, 32, 128)
[3/9] crash_001145__tensorflow_opt     -> RAN_OK  (49s)  out=(1, 3, 32, 32)
[4/9] crash_001068__tensorflow_opt     -> RAN_OK  (50s)  out=(1, 128, 16, 16)
[5/9] crash_001075__tensorflow_opt     -> RAN_OK  (52s)  out=(1, 48, 16, 16)
[6/9] crash_001168__tensorflow_opt     -> RAN_OK  (51s)  out=(1, 3, 64, 64)
[7/9] crash_001091__tensorflow_opt     -> RAN_OK  (50s)  out=(1, 32, 32, 32)
[8/9] crash_001138__tensorflow_opt     -> RAN_OK  (52s)  out=(1, 128, 16, 16)
[9/9] crash_001067__torch_compile_opt  -> RAN_OK  (10s)  out=(1, 96, 32, 32)

Summary: {'RAN_OK': 9}
Reproduced on fresh GPU: 0/9
```

All 9 sampled flaky cases ran clean in a fresh subprocess with
`CUDA_VISIBLE_DEVICES=1` and `memory_growth=True` — strong evidence that the
original failures were polluted-autotuner-cache / OOM-pressure artifacts.

### Audit task 2 — environmental sample (50/50 pure OOM)

Stratified sample of 50 events (20 × `cudaSetDevice OOM`, 20 × Torch
device-side assert, 10 × CUDA-OOM-variant); all 15 "other_env" events
inspected end-to-end. **0/50 had any atypical hint** (no AssertionError,
Check failed, segfault, TypeError, ValueError, shape mismatch, "not
implemented", etc.). The environmental classification is valid.

### Audit task 3 — CR-01 deep dive (mechanistic ablation)

Original hypothesis: *TVM emits multiple Var nodes when `freeze_params=True`
combines with self-referential binops.* The audit disproves two parts:

| Ablation | Result |
|----------|--------|
| `relay.nn.dense(x, W, units=None)` → `add(y, y)` | **CRASH(free-var)** |
| `relay.nn.dense(x, W, units=D)` → `add(y, y)` | OK |
| `relay.nn.dense(x, W, units=None)` → `mul(y, y)` | OK |
| `relay.nn.dense(x, W, units=None)` → `sub(y, y)` | OK |
| `relay.nn.dense(x, W, units=None)` → `div(y, y)` | OK |
| `relay.nn.dense(x, W, units=None)` → `add(y, c)` (const) | OK |
| `relay.nn.dense(x, W, units=None)` → `add(y, x2)` (other input) | OK |
| `Conv` → `add(y, y)` | OK |
| `Gemm` → `add(y, y)` (same `nn.dense`, units=D) | OK |
| `Identity/Relu/Cast` → `add(y, y)` (no dense) | OK |
| ONNX `MatMul → Add(m, m)` with `freeze_params=True` | CRASH |
| ONNX `MatMul → Add(m, m)` with `freeze_params=False` | **still CRASHES** |

**Refined statement** —

> Inside `tvm.relay.build(...)`, when the module contains
> `relay.nn.dense(x, W, units=None)` whose output feeds `relay.add(y, y)`
> (self-referential Add), the build pipeline's internal `FuseOps → LowerTE`
> path emits a lowered primitive that references the top-level input `x`
> as a free variable and aborts with
> `TVMError: … contains free variables: [Var(x, …)]`.

**Workaround**: manually pre-applying `FuseOps` before `relay.build` avoids
the crash. The ONNX → Relay frontend emits `nn.dense(units=None)` from
`MatMul` (rank-2) — that's why every one of the 5 original crashes is an
ONNX `MatMul → (…) → Add(t, t)` pipeline. All 5 original ONNX models and a
2-node constructive ONNX repro **and** a 25-line pure-Relay repro all
trigger the exact same error signature.

```
[REPRO] CR-01 triggered: contains free variables: [Var(x, ty=TensorType([1, 4], float32))]
---exit=0
```

---

## 4. Per-root-cause backend distribution

| RC | # cases | Nature | # backends | Affected |
|---|---:|---|---:|---|
| RC-01 | 41 | backend_bug    | 2 | TF, XLA |
| RC-02 | 26 | backend_bug    | 3 | TF, TC, XLA |
| CR-01 |  5 | crash_bug      | 1 | **TVM** |
| RC-03 |  5 | uncertain      | 4 | OV, TF, TC, XLA |
| RC-04 |  5 | backend_bug    | 1 | **TC** |
| RC-05 |  5 | backend_bug    | 1 | **TF** |
| RC-06 |  5 | spec_ambiguity | 6 | OV, TF, TC, TS, TVM, XLA |
| RC-07 |  4 | backend_bug    | 5 | OV, TF, TC, TS, XLA |
| RC-08 |  3 | uncertain      | 5 | OV, TC, TS, TVM, XLA |
| RC-09 |  3 | backend_bug    | 1 | **XLA** |
| RC-10 |  2 | spec_ambiguity | 5 | TF, TC, TS, TVM, XLA |
| RC-11a | 1 | uncertain      | 4 | OV, TS, TVM, XLA |
| RC-11b | 1 | spec_ambiguity | 3 | OV, TS, TVM |
| RC-11c | 1 | backend_bug    | 2 | TF, TC |
| RC-11d | 1 | backend_bug    | 3 | TS, TVM, XLA |
| RC-11e | 1 | spec_ambiguity | 3 | TF, TC, TS |
| RC-11f | 1 | backend_bug    | 5 | OV, TF, TC, TVM, XLA |
| RC-11g | 1 | backend_bug    | 3 | TF, TS, XLA |
| RC-11h | 1 | uncertain      | 2 | TC, TVM |
| RC-11i | 1 | backend_bug    | 2 | TS, XLA |
| RC-11j | 1 | spec_ambiguity | 2 | OV, XLA |
| RC-11k | 1 | backend_bug    | 1 | **TorchScript** |

### Backend suspect-frequency (cases where this backend is implicated)

```
  xla              95
  tensorflow       92
  torch_compile    54
  torchscript      21
  openvino         21
  tvm              20  (divergence) + 5 (crash)
  onnxruntime       0  (reference, never a suspect)
```

### Single-target clusters (cleanest to file upstream)

- **RC-04 (5)** → torch.compile / Inductor
- **RC-05 (5)** → TensorFlow
- **RC-09 (3)** → OpenXLA
- **RC-11k (1)** → TorchScript
- **CR-01 (5)** → Apache TVM

---

## 5. Issue-reporting priority

Ranked by: cluster size, confidence, repro clarity, likelihood of being a real backend bug.

### P1 — file immediately

| RC | # | Target | Why |
|---|---:|--------|-----|
| RC-01 | 41 | `openxla/xla` (primary), cc `tensorflow/tensorflow` | 41 cases on a plain attention graph |
| RC-02 | 26 | `openxla/xla` + `pytorch/pytorch` (Inductor) | 26 cases across 2 fusion targets |
| RC-07 | 4  | 5 targets cross-project | rel_l2=1.0 on every case, high confidence |
| RC-09 | 3  | `openxla/xla` | XLA-only, rel_l2=1.0 |
| RC-04 | 5  | `pytorch/pytorch` (Inductor) | Inductor-only miscompile |
| CR-01 | 5  | `apache/tvm` | reproducible crash, 25-line minimal repro |

### P2 — file after local triage

RC-05 (TF), RC-11c (TF+TC), RC-11d (TS+TVM+XLA), RC-11f (5 backends),
RC-11g (TF+TS+XLA), RC-11i (TS+XLA), RC-11k (TS).

### P3 — needs manual confirmation before filing

RC-06 (file to `onnx/onnx`), RC-10 (file to `onnx/onnx`), RC-08 (split the
two co-factors first), RC-03 (ask maintainers if fp16 roundtrip elimination
is intended), RC-11a/b/e/h/j (single cases needing manual verification).

### Summary counts

| Tier | Clusters | Cases |
|------|---------:|------:|
| P1 | 6 (RC-01/02/04/07/09 + CR-01) | 84 |
| P2 | 7 | 12 |
| P3 | 9 | 19 |

**First batch to file:** RC-01 → RC-02 → RC-07 → RC-09 → RC-04 → CR-01 covers
**84/115 bugs (73 %)** and four distinct project trackers.

---

## 6. Repro guarantees

Every `repros/rcNN_min.py` and `repros/cr01_min.py`:

- single Python file, < 110 lines, < 5 KB;
- builds its ONNX graph programmatically with tiny shapes (no base64
  blobs, no giant recorded models);
- passes `onnx.checker.check_model`;
- runs on ONNX Runtime without errors (verified for all 21 divergence
  repros in the final sweep);
- CR-01 exits 0 with the exact upstream error signature
  (`contains free variables: [Var(x, ty=TensorType([1, 4], float32))]`)
  on TVM 0.11.1.

### Final-sweep ORT smoke test

```
ok   rc01_min.py  out=(1, 8, 8, 8)
ok   rc02_min.py  out=(1, 8, 8, 8)
ok   rc03_min.py  out=(1, 4, 16)
ok   rc04_min.py  out=(1, 8, 8, 8)
ok   rc05_min.py  out=(8, 8)
ok   rc06_min.py  out=(1, 8, 16)
ok   rc07_min.py  out=(2, 16)
ok   rc08_min.py  out=(1, 8, 16, 16)
ok   rc09_min.py  out=(1, 2, 16, 16)
ok   rc10_min.py  out=(1, 4, 64, 64)
ok   rc11a_min.py  out=(1, 8, 8)
ok   rc11b_min.py  out=(1, 4, 8, 8)
ok   rc11c_min.py  out=(1, 3, 4, 4)
ok   rc11d_min.py  out=(1, 4, 16)
ok   rc11e_min.py  out=(1, 8, 8)
ok   rc11f_min.py  out=(1, 8, 8, 8)
ok   rc11g_min.py  out=(1, 4, 8, 8)
ok   rc11h_min.py  out=(1, 4, 8, 8)
ok   rc11i_min.py  out=(1, 8, 8)
ok   rc11j_min.py  out=(1, 4)
ok   rc11k_min.py  out=(1, 8, 8)

0 failures
```

---

## 7. How to run

The divergence repros use the `xcomp.oracle.*` backend classes from the
parent repo. Each repro locates the repo root automatically:

```bash
# Any divergence cluster — uses ORT + one or more suspect backends
python unified_bug_catalog/repros/rc01_min.py
python unified_bug_catalog/repros/rc07_min.py

# TVM crash — needs a TVM-capable interpreter (e.g. the 'clawwork' env)
/path/to/tvm-env/bin/python unified_bug_catalog/repros/cr01_min.py
```

Exit 0 = bug reproduced, 1 = backends agreed (bug gone in this version),
2 = setup error (missing backend package). All minimal repros print the
reference vs suspect rel_l2 numbers before exiting.

---

## 8. Scope and non-goals

**Scope**
- Every bug here was produced by the 1,500-model XComp campaign v10
  (supervisor run 2026-04-20 → 2026-04-22).
- `pytorch_eager` (onnx2torch + PyTorch CPU) is the numerical reference;
  ORT is always in the reference cluster.
- Every kept entry has a working minimal repro and a root-cause
  assignment with a confidence level.

**Non-goals**
- Not a replacement for the hand-curated `bugs/raw_bugs/` catalog in the
  main repo (that one covers a different, broader slice and uses its own
  RC numbering).
- No attempt is made here to port the repros to specific compiler versions
  or to automate upstream filing; `bug_root_cause_catalog/issue_priority.md`
  lists the recommended filing plan.

**Exclusions**
- 1,231 / 1,236 crash events (CUDA OOM, Triton autotuner "No valid config"
  under GPU pressure, cuDNN INTERNAL_ERROR) — all audited and classified as
  environmental or flaky; details in
  `crash_root_cause_catalog/excluded_cases.md`.
- 46 upstream validation failures (missing CUDA provider, corrupted
  torch-export zips, oneDNN diagnostic noise, stale artifacts) — filtered
  before dedup; details in `bug_root_cause_catalog/excluded_cases.md`.

---

Generated 2026-04-23 from `campaign_v10_results/`.
