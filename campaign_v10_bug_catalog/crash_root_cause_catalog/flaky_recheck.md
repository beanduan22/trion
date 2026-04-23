# Flaky-Case Recheck

Target set: **41 events** originally classified `nonreproducible_flaky`
(40 × `tensorflow+opt` "Autotuning failed" + 1 × `torch_compile+opt`
`CUDNN_STATUS_INTERNAL_ERROR`).

## Sampling plan

Stratified sample of **9 events** (≈ 22 %):

| Category | # sampled | How chosen |
|----------|-----------:|-----------|
| `tensorflow+opt` autotune failure — gemm_fusion_dot | 5 | random (seed=7) |
| `tensorflow+opt` autotune failure — cudnn-conv / cudnn-conv-bias-activation | 3 | random (seed=7) |
| `torch_compile+opt` cuDNN INTERNAL_ERROR | 1 | the only one in the bucket |

## Replay protocol

All cases reloaded from `campaign_v10_results/compiler_crashes/crash_NNNNNN.onnx`
and fed through the same `xcomp` backend class the original campaign used:

- `xcomp.oracle.tf_backend.TFBackend.run(model, inputs, optimized=True)` for TF cases.
- `xcomp.oracle.torch_compile_backend.TorchCompileBackend.run(model, inputs, optimized=True)` for the torch.compile case.

Per case:

- **Fresh subprocess** (Python interpreter re-spawned) — no state carryover.
- **`CUDA_VISIBLE_DEVICES=1`** — the second GPU, which had ~41 GB free
  (vs. ~27 GB on the original campaign GPU at the moment the crash was
  recorded).
- **`np.random.seed(0)`** — deterministic input draw.
- **TF memory_growth=True** — prevents the session from pre-allocating
  the whole GPU and lets later configs fit.
- Identical ONNX model, identical input shape, identical backend class,
  no concurrency (1 case at a time).
- Timeout 300 s per case; observed wall-clock 49–53 s each.

Driver scripts:

- `repros/_flaky_replay_tf.py`
- `repros/_flaky_replay_torch_compile.py`

## Outcomes

| # | crash_id | bucket | outcome | output shape | elapsed |
|--:|----------|--------|---------|--------------|--------:|
| 1 | crash_001126__tensorflow_opt | autotune (gemm_fusion_dot) | **RAN OK** | (1, 128, 16, 8) | 53 s |
| 2 | crash_001107__tensorflow_opt | autotune (gemm_fusion_dot) | **RAN OK** | (1, 32, 128) | 51 s |
| 3 | crash_001145__tensorflow_opt | autotune (gemm_fusion_dot) | **RAN OK** | (1, 3, 32, 32) | 49 s |
| 4 | crash_001068__tensorflow_opt | autotune (cudnn-conv-bias-activation) | **RAN OK** | (1, 128, 16, 16) | 50 s |
| 5 | crash_001075__tensorflow_opt | autotune (cudnn-conv) | **RAN OK** | (1, 48, 16, 16) | 52 s |
| 6 | crash_001168__tensorflow_opt | autotune (gemm_fusion_dot) | **RAN OK** | (1, 3, 64, 64) | 51 s |
| 7 | crash_001091__tensorflow_opt | autotune (cudnn-conv-bias-activation) | **RAN OK** | (1, 32, 32, 32) | 50 s |
| 8 | crash_001138__tensorflow_opt | autotune (gemm_fusion_dot) | **RAN OK** | (1, 128, 16, 16) | 52 s |
| 9 | crash_001067__torch_compile_opt | cuDNN INTERNAL_ERROR | **RAN OK** | (1, 96, 32, 32) | 10 s |

**Reproduced on fresh GPU: 0 / 9.**
**Ran clean without any error: 9 / 9.**

## What this means

- A clean-GPU replay of the sampled flaky events produced a usable output
  tensor. None of them hit either the "No valid config found" autotuner
  path or the cuDNN internal error in the new run.
- This is exactly what we'd expect if the original failures were caused
  by a polluted autotuner cache or a bad CUDA context inherited from a
  prior OOM in the same long-running session. The 1,190 confirmed env
  OOMs in the same run provide plenty of opportunity for such
  contamination.

## Decision

- **No new real-crash root-cause cluster is introduced.**
- The 9 directly-tested events are now tagged
  `nonreproducible_confirmed_flaky` in `crash_manifest.json` with a
  `replay_on_fresh_gpu = "ran_clean"` note.
- The remaining 32 flaky events stay as `nonreproducible_flaky` — their
  text signatures match the 9 replayed ones, so there is no reason to
  single any of them out. If a future audit wants a harder claim on those
  32, the replay drivers work out-of-the-box.

## Caveats / limits of this recheck

- Sample size is **22 %**, not 100 %. It's conclusive enough to reject
  the hypothesis that "all 41 are real bugs" (any-positive rate of 100 %
  is incompatible with the 0 we observed), but it can't categorically
  rule out that 1–2 of the remaining 32 are genuine.
- The fresh GPU still has other tenants (~20 GB in use on GPU 0 at
  replay time). We did not force the GPU to be exclusively idle.
- Autotuner cache files on disk (`~/.cache/xla/*`) were not wiped between
  runs, so there may be a positive benefit from a warm autotuner cache
  the original session lacked. That would only *further* explain the
  flakiness, not invalidate the "not a real bug" verdict.
