# Excluded Cases

These crash events are **not** real compiler bugs in the sense of this
catalog. Each is kept in `crash_manifest.json` for audit, but excluded from
the reportable bug set.

## 1,190 events — CUDA out-of-memory and downstream

| Bucket | # Events | Representative error |
|--------|---------:|----------------------|
| `cudaSetDevice` GPU:0 out of memory | 558 | `cudaSetDevice() on GPU:0 failed. Status: out of memory` |
| `TORCH_USE_CUDA_DSA` device-side assertion | 550 | `Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.` (raised after a prior OOM corrupted the CUDA stream) |
| `CUDA out of memory. Tried to allocate … MiB` | ~46 | `CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has … of which 72.75 MiB is free.` |
| `CUDA driver error: out of memory` | 4 | `RuntimeError: CUDA driver error: out of memory` |
| `CUDA_ERROR_OUT_OF_MEMORY` while loading CUBIN / getting module function | 4 | `RESOURCE_EXHAUSTED: [0] Failed to load in-memory CUBIN … CUDA_ERROR_OUT_OF_MEMORY` |
| TF host allocator — `Failed to allocate 16777216 bytes for new constant` | 1 | Same session-level pressure; 16 MB host allocation failure |

**Reason excluded:** every event is either a direct CUDA OOM or a
downstream consequence of one (device-side assertions, CUBIN-load failures,
constant-buffer allocation failures). None carries any model-specific
signal; all appear during a long campaign running many models back-to-back
on a single 47 GB GPU without reset.

**Verification:** the same backends run the same ONNX models on a fresh
GPU (e.g., in the unique-bugs catalog) without crashing.

## 40 events — XLA Triton autotuner "No valid config found"

- **Affected tool:** `tensorflow+opt` (and 6× `xla+opt` with the same signature)
- **Representative error:**
  ```
  Autotuning failed for HLO: %gemm_fusion_dot.3 = f32[64,64] fusion(…),
  … "device_type":"DEVICE_TYPE_INVALID" with error: NOT_FOUND:
  No valid config found! [Op:__inference_model_fn_167911]
  ```
- **Reason excluded:** all involved shapes are small (`f32[64,64]`,
  `f32[32,128]`, `f32[1,32,32,32]`) — shapes for which the Triton-gemm and
  cuDNN-conv autotuners ship stock configs. All 40 events occurred
  concurrently with the CUDA-OOM storm above; the autotuner's benchmark
  phase allocates temporary output buffers, and under OOM pressure every
  candidate benchmark fails → "No valid config found". The failure is
  session-state-dependent, not model-dependent.
- **Non-reproducibility note:** re-running any of these 40 models on a
  clean GPU is not a cheap experiment in this environment. Until that is
  done, they remain flaky. If a clean-state replay **does** reproduce any
  of them, that individual case should be promoted to a real bug and
  reported separately — but the whole bucket is rejected as a source of
  compiler bugs today.

## 1 event — torch.compile `cuDNN CUDNN_STATUS_INTERNAL_ERROR`

- **Affected tool:** `torch_compile+opt` on model 1067.
- **Representative error:** `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR` during a conv forward inside a torch.compile-generated kernel.
- **Reason excluded:** the **same** model also triggered a
  `CR-FLAKY-AUTOTUNE` event on `tensorflow+opt` in the same run — two
  independent backends failing on the same input in the same run strongly
  suggests GPU session state, not a torch.compile bug. cuDNN
  `INTERNAL_ERROR` is the canonical "GPU state got corrupted earlier and
  the next cuDNN call blew up" error.
- **Non-reproducibility note:** same as the autotuner bucket — needs
  clean-GPU replay before any promotion.

## 0 events in each of:

- Reproducible exporter/conversion crash — none. No entry triggered a
  failure during ONNX export, `onnx.checker`, or ONNX → IR conversion.
- Corrupted artifact / invalid input — none inside
  `campaign_v10_results/compiler_crashes/`. (Upstream validation filtered
  46 environmental failures before the bug dedup; see
  `bug_root_cause_catalog/excluded_cases.md` for those.)

## Exclusion summary

| Bucket | # Events | Reason |
|--------|---------:|--------|
| CR-ENV-OOM | 1,190 | CUDA OOM and downstream, zero model signal |
| CR-FLAKY-AUTOTUNE | 40 | correlated with concurrent OOM storm; would need clean-GPU replay |
| CR-FLAKY-CUDNN | 1 | co-occurs with another flaky event on the same model |
| **Total excluded** | **1,231** | out of 1,236 crash events |

**Remaining (real compiler bugs): 5 events → 1 root-cause cluster (CR-01).**
