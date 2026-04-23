# Environmental-Case Sample Audit

Target set: the **1,190 events** classified `environment` in the initial
catalog.

## Method

No rerun — text inspection only. For each sampled event the full error
text is loaded from `crash_*_report.json` and scanned for **any** line
that is *not* an OOM / CUDA-state line. Specifically every line is
matched against the pattern

```
AssertionError | Check failed | segfault | TypeError | ValueError
 | internal assertion | not implemented | shape mismatch
 | invalid graph | abort | not supported
```

A case is flagged `POSSIBLY_REAL` if it contains any matching line and
`OOM_only` otherwise.

## Stratified sample

| Bucket | Population | Sampled |
|--------|-----------:|--------:|
| direct `cudaSetDevice out of memory`                      | 558 | 20 |
| PyTorch device-side assert (`TORCH_USE_CUDA_DSA` or "CUDA kernel errors might be asynchronously reported") | 550 | 20 |
| `CUDA out of memory. Tried to allocate … MiB`             | 54 | 10 |
| `cuda_driver_OOM` (`RuntimeError: CUDA driver error: out of memory`) | 4 | — (included in above) |
| **Total stratified sample**                               | — | **50** |

In addition, all **15** remaining `other_env` events were inspected
exhaustively — they turned out to be TF graph-executor
`Out of memory while trying to allocate 16.00 MiB …`, also pure OOM
under a different wording.

## Results

| | Count |
|-|------:|
| Sampled events flagged `OOM_only` | **50 / 50** |
| Sampled events flagged `POSSIBLY_REAL` | **0 / 50** |
| `other_env` events fully inspected | 15 / 15 |
| Of those, POSSIBLY_REAL | 0 / 15 |

Every sampled event's first non-empty error line **and** last non-empty
error line is a CUDA-allocator or TF-allocator out-of-memory message.
There is no trailing stack below the OOM that hints at a real bug, and
there is no leading pre-amble with a compiler-side error above the OOM.

## Representative samples (one per bucket)

- `crash_000499__tensorflow_opt` — `cudaSetDevice() on GPU:0 failed. Status: out of memory`
- `crash_000426__torch_compile_opt` — `RuntimeError: CUDA driver error: out of memory`
- `crash_001027__tensorflow_opt` — `cudaSetDevice() on GPU:0 failed. Status: out of memory`
- `crash_001113__tensorflow_opt` (the `other_env` bucket) — `Out of memory while trying to allocate 16.00MiB. [Op:__inference_model_fn_36324]`

## Decision

- **The environment bucket is validated.** No sampled event is hiding a
  real compiler bug under an OOM banner, and the remaining 15 "other"
  events are just OOM under a different message.
- The 1,190 `environment` classifications in `crash_manifest.json` stand
  unchanged.

## Caveats

- This audit inspects text only; it does not rerun any environmental
  case. Rerunning 1,190 CUDA-OOM events on a fresh GPU is not a useful
  experiment — they will either OOM again (if that shape is actually
  large enough to blow the GPU) or not, depending on the free-memory
  snapshot, which isn't model-relevant.
- The pattern list used for `POSSIBLY_REAL` flagging is intentionally
  broad but not exhaustive. If a compiler emits an unusual phrase we did
  not include, a case could slip through. The "first/last line is OOM"
  check for every sampled case provides a second line of defense
  against that.
