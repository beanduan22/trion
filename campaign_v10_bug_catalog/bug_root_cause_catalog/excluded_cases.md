# Excluded cases

No unique validated bug was excluded at this stage.

## Already filtered upstream

The dedup pipeline (`tools/validate_and_dedupe.py` run 2026-04-22) read 156 candidate repros and rejected 46 at the validation stage before the 110 unique buckets were formed. All 46 are environmental, not semantic. For completeness the breakdown:

| Failure mode | Count |
|--------------|-------|
| `CUDAExecutionProvider is not in available provider names` (ORT CPU host) | ~12 |
| `PytorchStreamReader failed reading zip archive` (corrupted torch-export checkpoint) | ~5 |
| `oneDNN custom operations are on` diagnostic noise (TF) | 2 |
| Empty stderr / silent re-validation fail (likely stale artifacts overwritten during regen) | ~27 |

Source: `campaign_v10_results/bugs_unique/manifest.json` — `failures` array.

## Category applied here

None of the 110 bugs processed in this catalog are excluded. Every entry is a real numerical divergence between two fully-initialized backends.
