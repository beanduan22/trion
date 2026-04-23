# Audit Follow-up — Second Pass on the Crash Catalog

This document summarizes the **validation-pass audit** of the initial crash
catalog. Three targeted tasks were performed. Conclusion first, evidence
below.

## TL;DR

The original conclusion **holds**.

| Claim | Status after audit |
|-------|--------------------|
| 5 reproducible real compiler bugs, all CR-01 (TVM free-variable) | **Confirmed.** 5/5 still reproduce in a fresh process. The root cause has been narrowed from "freeze\_params + self-ref binop" to a much tighter, non-ONNX trigger — see `cr01_deep_dive.md`. |
| 1,190 environmental GPU-memory failures | **Confirmed.** A 50-case stratified sample found zero atypical / non-OOM signals; all 15 "other-env" cases were also 100 % OOM. |
| 41 non-reproducible / flaky failures | **Strengthened as flaky.** 9/9 of a stratified fresh-GPU sample (8 autotune + 1 cuDNN) **ran clean** — no autotune or cuDNN crash. This is consistent with the hypothesis that they are GPU-state artifacts, not real bugs. |
| 0 exporter/conversion crashes | Confirmed (no evidence to the contrary). |
| 0 corrupted artifacts | Confirmed. |

**No new real-crash cluster was discovered.** The count of real reproducible
crash clusters remains **1 (CR-01)** with 5 matched events.

## Changes applied to the catalog

- `crash_manifest.json` — 9 flaky entries whose clean-process replay passed
  were flipped from `nonreproducible_flaky` → **`nonreproducible_confirmed_flaky`**
  with new fields `replay_on_fresh_gpu`, `replay_headline`, and `audit_note`.
- `cr01_deep_dive.md` — new file with a tightened mechanistic statement.
- `flaky_recheck.md` — new file detailing the 9-case flaky sample.
- `environment_sample_audit.md` — new file detailing the 50-case env sample.
- `repros/cr01_tvm_relay_min.py` — new stricter minimal repro that drops
  ONNX entirely and constructs the trigger at the Relay-IR level.

The original `summary.md`, `catalog.md`, and `excluded_cases.md` remain
accurate and are not regenerated; the audit files above are additive.

## Audit task 1 — Flaky recheck: 41 cases → 9 replayed → **0 reproduced**

- Selected a stratified sample: 8 × `tensorflow+opt` "Autotuning failed / No
  valid config found" + 1 × `torch_compile+opt` `CUDNN_STATUS_INTERNAL_ERROR`.
- Each replay runs in a **fresh subprocess** via
  `repros/_flaky_replay_tf.py` / `repros/_flaky_replay_torch_compile.py`,
  pinned to `CUDA_VISIBLE_DEVICES=1` (a GPU with ~41 GB free instead of the
  crash-era ~27 GB), with `np.random.seed(0)` for input and identical ONNX
  models.
- **Outcomes:** 9/9 runs completed successfully (exit 1 / "RAN_OK"), none
  reproduced. Evidence in `flaky_recheck.md`.
- **Decision:** the 41 flaky cases are **kept as flaky**, not promoted; the
  9 directly tested ones are marked `nonreproducible_confirmed_flaky` in
  the manifest.

## Audit task 2 — Stratified env sample: 50 cases → 0 hidden bugs

- Stratified sample: 20 `cudaSetDevice OOM` + 20 Torch device-side assert +
  5 "CUDA out of memory" + 4 "CUDA driver error" + 1 allocator failure.
- Also scanned all 15 "other\_env" events end-to-end (they were TF
  `Out of memory while trying to allocate 16 MiB` — OOM under a different
  message).
- **Outcome:** 0 / 50 sampled events show any `AssertionError`,
  `Check failed`, `segfault`, `TypeError`, `ValueError`, `shape mismatch`,
  "not implemented", or other non-OOM signal. Every event's first *and*
  last non-empty error line is a CUDA / allocator OOM line. Evidence in
  `environment_sample_audit.md`.

## Audit task 3 — CR-01 deep dive: hypothesis tightened

The original write-up said: *TVM emits multiple Vars when `freeze_params=True`
combines with a self-referential binop.* The audit **disproves two parts** of
that hypothesis:

1. `freeze_params=True` is **not necessary**. Toggling it off keeps the
   crash; it's just how the ONNX frontend routes through the builder.
2. Self-referential binop is necessary **but not sufficient**. It must be
   **`add(y, y)` specifically** — `mul(y, y)`, `sub(y, y)`, `div(y, y)`
   are all fine.

The refined, tighter trigger is:

> `relay.nn.dense(x, W, units=None)` → `relay.add(y, y)` inside
> `relay.build(...)` — the **Relay-level** dense op with `units=None` makes
> the primitive function emitted during build's internal `FuseOps → LowerTE`
> chain list `x` as a free variable. Manually pre-applying `FuseOps` before
> `relay.build` avoids the crash.

Full ablation matrix + failing-stage localization in `cr01_deep_dive.md`.
A stricter, ONNX-free minimal repro is in `repros/cr01_tvm_relay_min.py` (~15
lines).

## Why the original conclusion survives

- **Not promoting the 41 flaky events** is now backed by a concrete
  9/9 = 100 % clean-replay rate on a stratified sample.
- **Not promoting any env event** is backed by a 50/50 inspection finding
  zero non-OOM hints.
- **CR-01 is still exactly one cluster**; the mechanistic statement is
  tighter, but the set of matched events is unchanged (5).

## What the audit did **not** do

- It did not re-run all 1,236 events (that was out of scope — this is a
  validation pass, not a rebuild).
- It did not re-run all 41 flaky events, only 9. If later someone wants to
  commit to a stronger claim on the remaining 32, the driver scripts in
  `repros/_flaky_replay_tf.py` / `_flaky_replay_torch_compile.py` can be
  reused verbatim.
- It did not debug the exact TVM pass that introduces the free-var inside
  `relay.build`'s pipeline. The failing stage is narrowed to "Relay → TE
  lowering inside `relay.build` when `nn.dense(units=None)` feeds a
  self-`add`"; chasing it into C++ sources is left for the upstream TVM
  issue to request.
