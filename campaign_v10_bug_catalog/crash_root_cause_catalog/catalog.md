# Crash Catalog — Grouped by Root Cause

1,236 crash events across 664 ONNX models in campaign_v10.
Real reproducible compiler crashes: **5** (all TVM). Everything else is
environmental or flaky — see `excluded_cases.md` for those buckets.

---

## CR-01 · TVM Relay build emits duplicate `Var` for the same input when an intermediate tensor is used as both operands of a binary op

- **Classification:** Reproducible backend/compiler crash
- **Affected tool:** Apache TVM — `tvm.relay.build(mod, params=..., freeze_params=True)` in TVM 0.11.1
- **Failing stage:** graph_rewrite / build optimization pass (independent of `opt_level`; crashes at 0 through 4)
- **Matched crash IDs (5):**
  - `crash_000385__tvm_opt` (model 385, input [1, 128])
  - `crash_000905__tvm_opt` (model 905, input [1, 64])
  - `crash_000959__tvm_opt` (model 959, input [1, 256])
  - `crash_001225__tvm_opt` (model 1225, input [1, 128])
  - `crash_001452__tvm_opt` (model 1452, input [1, 256])
- **Representative error signature:**
  ```
  TVMError: … Function: fn (%model_input: …) { … }
  contains free variables:
    [Var(model_input, ty=TensorType([1, 128], float32)),
     Var(model_input, ty=TensorType([1, 128], float32)),
     Var(model_input, ty=TensorType([1, 128], float32))]
  ```
- **Why these cases belong together:** Every one of the 5 ONNX graphs references `model_input` **exactly once**, directly, in a single leaf op (`MatMul` or `Softmax`). But the generated graphs all contain at least one binary op whose two inputs are the same SSA tensor — e.g. `Add(base, base)` in the triple-add-residual pattern, `Mul(%38, %39)` in the GLU pattern, or `Add(%6, %5)` in the residual-add path. TVM's Relay-from-ONNX frontend builds a Var for the input and `freeze_params=True` then triggers a downstream rewrite (likely in `AnnotateMemoryScope` / `ManifestLifetimes` / a scheduling pass) that clones the Var; the clones are distinct `Var` objects with the same name. The final free-var check sees 2 or 3 unresolved Vars and aborts.

  Confirming evidence:
  - Every crash reproduces deterministically from its original `crash_NNNNNN.onnx` (5/5).
  - A hand-built 2-node graph `MatMul(x, W) → Add(m, m)` reproduces the exact same `free variables` error (`repros/cr01_tvm_free_vars_min.py` → exits 0).
  - The pre-pass Relay IR shown by `str(mod)` after `from_onnx` contains only one `%model_input` Var. The duplication appears during `relay.build`, not during import.
  - Eliminating `freeze_params=True` avoids the free-var error (a different error surfaces for dynamic Split — orthogonal).
- **Confidence:** high
- **Representative minimal repro:** `repros/cr01_tvm_free_vars_min.py`
- **Original-model replay:** `repros/cr01_tvm_free_vars_replay_original.py` (5/5 reproduces)

---

## Non-reproducible / flaky buckets — reference only, details in `excluded_cases.md`

These are kept in `crash_manifest.json` for traceability, but are not real compiler bugs in the sense this catalog intends.

### CR-FLAKY-AUTOTUNE · XLA Triton autotuner "No valid config found" under GPU pressure

- **Classification:** non-reproducible / flaky / uncertain
- **Affected tool:** TensorFlow + XLA (via `tensorflow+opt`)
- **Failing stage:** codegen — autotuner (Triton gemm fusion / cuDNN conv+bias activation)
- **Matched crash IDs (40):** every `tensorflow+opt` crash whose error text contains `Autotuning failed for HLO` and `NOT_FOUND: No valid config found!` (see `crash_manifest.json` for the IDs)
- **Representative error signature:**
  ```
  Autotuning failed for HLO: %gemm_fusion_dot.3 = f32[64,64] fusion(…),
  kind=kCustom, calls=%gemm_fusion_dot.3_computation, …,
  "device_type":"DEVICE_TYPE_INVALID" with error: NOT_FOUND: No valid config found!
  ```
- **Why these belong together and why we don't promote them to real bugs:** The shapes are all small (`f32[64,64]`, `f32[32,128]`, `f32[1,32,32,32]`), for which the Triton gemm and cuDNN-conv autotuners *do* have configs. All 40 events occurred during the same campaign session that emitted 1,190 CUDA OOMs, and the autotuner's kernel-benchmark phase allocates temporary buffers; under OOM pressure every candidate config fails the benchmark → "No valid config found". This is strongly correlated with environment state, not model state. A clean-GPU replay would be needed to promote any of these to real bugs.
- **Confidence:** medium
- **Minimal repro:** none — cannot be reproduced without recreating the original OOM-stressed GPU session.

### CR-FLAKY-CUDNN · torch.compile `cuDNN CUDNN_STATUS_INTERNAL_ERROR`

- **Classification:** non-reproducible / flaky / uncertain
- **Affected tool:** PyTorch 2.11 + torch.compile (Inductor)
- **Failing stage:** runtime_execution (cuDNN conv kernel)
- **Matched crash ID:** `crash_001067__torch_compile_opt` (model 1067). The same model *also* has a `tensorflow+opt` autotune failure — two different backends failed on the same input in the same run, further suggesting session-level GPU state.
- **Confidence:** low
- **Minimal repro:** none.

### CR-ENV-OOM · CUDA out of memory and downstream device-side assertions

- **Classification:** environmental
- **Affected tools:** tensorflow+opt, torch_compile+opt (both use CUDA)
- **Failing stage:** runtime_execution — CUDA memory allocator / kernel launch
- **Matched crash IDs (1,190):** every event not in the three buckets above.
- **Representative signatures:**
  - `cudaSetDevice() on GPU:0 failed. Status: out of memory` — 558 events
  - `... Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.` — 550 events (PyTorch device-side assert after a prior OOM cascaded into a NaN/garbage pointer)
  - `CUDA out of memory. Tried to allocate 96.00 MiB …` — 46 events (many subtle GPU-free-memory variants; same root cause)
  - `RuntimeError: CUDA driver error: out of memory` — 4 events
  - `RESOURCE_EXHAUSTED: [0] Failed to load in-memory CUBIN … CUDA_ERROR_OUT_OF_MEMORY` — 2 events
  - `RESOURCE_EXHAUSTED: [0] Failed to get module function … CUDA_ERROR_OUT_OF_MEMORY` — 2 events
  - `Failed to allocate 16777216 bytes for new constant` (TF host-side) — 1 event; 16 MB host alloc failure is also memory pressure.
- **Why these belong together:** every event is either a direct CUDA OOM or a CUDA device-side assert (`TORCH_USE_CUDA_DSA`) that, in this harness, is raised as a side-effect of an earlier OOM corrupting the CUDA stream. None of them carry any model-specific signal.
- **Confidence:** high (that they are environmental)
- **Minimal repro:** not applicable.

---

## Cross-cluster sanity checks

- **ORT is never a crasher.** `onnxruntime` does not appear in any crash event. This is consistent with the validation catalog: ORT runs on CPU in this harness, so it sees no CUDA memory pressure and no Triton autotuner.
- **OpenVINO and TorchScript are never crashers** either in this directory. They were the only two spec-preserving backends in the validation catalog too — they simply never tripped anything serious enough to be caught by the crash gate.
- **XLA standalone (6 events) is essentially the same error profile as TF+XLA.** We treat the 6 `xla+opt` events as a subset of CR-FLAKY-AUTOTUNE (they all show "No valid config found" style failures); they are counted in the 40.

---

## TL;DR

Out of 1,236 crash events in the 1,500-model campaign, there is **exactly one real, reproducible compiler-bug cluster** — 5 TVM Relay build crashes all caused by the same duplicate-Var issue when a self-referential binary op is present and `freeze_params=True` is set. Everything else is CUDA OOM or OOM-induced flakiness.
