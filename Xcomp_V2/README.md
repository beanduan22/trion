# Xcomp_V2

This folder contains a refined copy of the original `xcomp` implementation as
the importable package `xcomp_v2`.

V2 keeps the existing pattern library, including compatibility-cache filtering.
It does not impose an arbitrary pattern-count cap. Instead, the library reports
how many patterns were retained, excluded, and known-divergent for the active
backend set.

Key refinements:

- Bigram credit now affects sampling instead of only collecting pair counters.
- No-opt backend crashes are recorded as crash events.
- `save_all=True` writes non-bug models under `all_models/` instead of naming
  them `bug_*.onnx`.
- The frontend spec gate no longer falls back to `onnx2torch` when ORT no-opt is
  unavailable.
- The copied package uses `xcomp_v2` imports so it can coexist with `xcomp`.

Example:

```bash
cd /home/binduan/myspace/trion/Xcomp_V2
python run_xcomp_v2.py
```

---

## 9-Backend Smoke Probe and UpSet Plot

This section documents how to (a) run the cross-compiler smoke probe that
classifies every pattern on every backend, and (b) draw a clean UpSet figure
that summarises which backends co-fail.

### 1. What the smoke probe does

`tools/full_probe_9.py` instantiates **every pattern** in the live OTP library
on **every backend** in `ALL_BACKENDS`. The 9 backends are:

| Group              | Backends                                                                 |
| ------------------ | ------------------------------------------------------------------------ |
| In-process         | `onnxruntime`, `torchscript`, `xla`                                      |
| Subprocess workers | `torch_compile`, `tvm`, `tensorflow`, `openvino`, `tflite`, `tensorrt`   |

For each `(pattern, backend)` cell it picks the first compatible structural
seed from a small fixed list, builds a single-pattern ONNX model, runs it with
`optimized=True`, and records one of:

| Status         | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `pass`         | Backend ran the model and produced a numerically-valid output |
| `fail`         | Backend ran but reported `r.ok == False`                      |
| `crash`        | Backend raised, segfaulted, or its worker died                |
| `unavailable`  | Backend reported `is_available() == False` at probe time      |
| `incompatible` | No seed in `_SEEDS` satisfied `pat.is_compatible`             |

Subprocess backends each get a long-lived helper process so a single crash
does not lose the rest of the run. CPU-only backends (`tensorflow`, `tflite`)
are spawned with `CUDA_VISIBLE_DEVICES=""` so their bundled CUDA libs do not
clash with the GPU stack used by torch/TVM.

Progress is checkpointed to disk every 50 patterns; the run can be killed and
resumed at any time.

### 2. Run the probe

```bash
cd /home/binduan/myspace/trion/Xcomp_V2

# Full probe, ~2-3 hours unattended
nohup python tools/full_probe_9.py \
    > logs/full_probe_9.log 2>&1 &

# Re-probe a single backend without losing the others
FORCE_REPROBE=1 python tools/full_probe_9.py
```

Outputs:

```
campaign_v10_results/
  full_probe_9.progress.json   # checkpoint, written every 50 patterns
  pattern_compat_v10_9b.json   # final atomic snapshot (== progress at end)
```

Both files share the same schema:

```jsonc
{
  "backends": ["onnxruntime", "torchscript", ..., "tensorrt"],
  "patterns": {
    "<pattern-name>": {
      "<backend>": { "status": "pass" | "fail" | ..., "detail": "..." },
      ...
    },
    ...
  },
  "safe_patterns": ["..."],         // pass on every live backend
  "summary": { "full_probe_at": "...", "wall_time_s": 12345.6 }
}
```

### 3. Latest run snapshot (883 patterns)

| backend         | pass | fail | crash | incompat |
| --------------- | ---: | ---: | ----: | -------: |
| `onnxruntime`   |  860 |    9 |     0 |       14 |
| `tensorrt`      |  865 |    4 |     0 |       14 |
| `tvm`           |  862 |    7 |     0 |       14 |
| `openvino`      |  856 |   13 |     0 |       14 |
| `torchscript`   |  795 |   74 |     0 |       14 |
| `torch_compile` |  795 |   74 |     0 |       14 |
| `xla`           |  734 |  135 |     0 |       14 |
| `tensorflow`    |  692 |  177 |     0 |       14 |
| `tflite`        |  671 |  198 |     0 |       14 |

Aggregate: **`safe_patterns = 561`** (pass on all 9), **234 patterns fail on
at least one backend**, and only **16 distinct fail-signatures** appear across
those 234 patterns.

### 4. Draw the UpSet figure

```bash
python tools/plot_upset_9b.py
# optional flags:
#   --src        path to a probe JSON (default: full_probe_9.progress.json)
#   --out-dir    where to write figures (default: campaign_v10_results/figures)
#   --min-size N hide intersections smaller than N (default 1)
```

Outputs (PNG, 160 dpi):

```
campaign_v10_results/figures/
  fail_upset.png                 # bug-hunting view
  pass_upset.png                 # coverage view
  pattern_status_indicators.csv  # raw bool table for re-plotting
```

The CSV has one row per pattern and two bool columns per backend
(`<backend>__fail`, `<backend>__pass`), so you can rebuild any chart in
seaborn / R / Excel without re-parsing JSON.

### 5. How to read the UpSet plot

An UpSet plot is the modern replacement for a 9-set Venn diagram. It has
three coupled panels:

```
                  ┌──────────────────────────────────────┐
   Top bar chart  │  count of patterns in this           │
   (cardinality)  │  intersection                        │
                  └──────────────────────────────────────┘
                       ●   ●   ●           ●              ← matrix:
                  ●─●  │   │   │           │                filled+linked
                  │    ●   ●   ●─●─●       ●─●              dots = the
                  │    │   │   │ │ │       │ │              backends
                  │    │   │   │ │ │       │ │              participating
                  ●    ●   ●   ● ● ●       ● ●              in this column
                  ┌──────────────────────────────────────┐
   Left bar chart │  count of patterns where this        │
   (set size)     │  backend appears at all              │
                  └──────────────────────────────────────┘
```

- **Each column = one intersection** (one specific combination of backends).
- **Top bar = how many patterns share that exact failure pattern.**
- **Filled & connected dots in the matrix tell you *which* backends.**
- **Left bar = total per backend** (e.g. `tflite=198` for the fail plot).
- Columns are ordered by cardinality (largest left), so the dominant
  failure modes are immediately visible.

For the **fail plot**:
- A column with dots on `tensorflow` + `tflite` + `xla` and a top bar of
  `84` means: 84 patterns fail on exactly those three backends and pass on
  the other six.
- A column with a single dot on `onnxruntime` is the holy grail of
  divergence testing: 8 backends agree the pattern is fine, ORT alone
  disagrees → likely an ORT bug.

### 6. The 16 fail-signatures in the current run

| #  | Failing backends                                                              | Patterns | Interpretation                                              |
| -: | ----------------------------------------------------------------------------- | -------: | ----------------------------------------------------------- |
|  1 | `xla` + `tensorflow` + `tflite`                                               |       84 | Shared TF op-coverage / spec gap                            |
|  2 | `tensorflow` + `tflite`                                                       |       43 | TF-only spec gap (XLA happens to fold it)                   |
|  3 | `torchscript` + `torch_compile` + `xla` + `tensorflow` + `tflite`             |       37 | ONNX op missing in both Torch frontends *and* the TF stack  |
|  4 | `torchscript` + `torch_compile`                                               |       22 | onnx2torch / torch.onnx unsupported op                      |
|  5 | `tflite`                                                                      |       18 | TFLite-only quantization edge                               |
|  6 | `openvino` + `torchscript` + `torch_compile` + `xla` + `tensorflow` + `tflite`|       10 | Wide unsupported-op cluster, ORT/TVM/TRT clean              |
|  7 | `onnxruntime`                                                                 |        5 | **Candidate ORT bugs**                                      |
|  8 | `tvm`                                                                         |        5 | **Candidate TVM bugs**                                      |
|  9 | `torchscript` + `torch_compile` + `tflite`                                    |        2 |                                                             |
| 10 | `onnxruntime` + `tensorrt`                                                    |        2 | Likely shared ORT/TRT decomposition issue                   |
| 11 | `tvm` + `openvino` + `torchscript` + `torch_compile`                          |        1 |                                                             |
| 12 | `xla` + `tflite`                                                              |        1 |                                                             |
|  … | (4 more singletons)                                                           |        4 |                                                             |

Two takeaways for the campaign:

1. The TF/TFLite/XLA cluster (~174 patterns, rows 1-3) is **one** shared
   op-coverage gap, not three independent ones — most of it traces back to
   the same TF op-set / quantization handling. Merging these into a single
   "TF stack" track avoids re-finding the same bug.
2. The lone-backend signatures (rows 5, 7, 8, plus the singletons) are the
   **high-value divergence candidates**. Every other backend agrees, so the
   disagreeing one is almost certainly carrying a real bug. Triage those
   first.

### 7. Customising the plot

The default figure shows every non-empty intersection. For a more compact,
publication-ready chart:

```bash
# hide the long tail (anything ≤4 patterns)
python tools/plot_upset_9b.py --min-size 5

# point at a different probe (e.g. an older snapshot for diffing)
python tools/plot_upset_9b.py \
    --src campaign_v10_results/pattern_compat_v10.json \
    --out-dir campaign_v10_results/figures_old
```

Knobs you may want to edit inside `tools/plot_upset_9b.py`:

- `BACKENDS` order — left-bar order in the figure. Currently grouped as
  *runtime engines* first (`onnxruntime`, `tensorrt`, `tvm`, `openvino`),
  then *PyTorch frontends* (`torchscript`, `torch_compile`), then *TF stack*
  (`xla`, `tensorflow`, `tflite`). This keeps related failures visually
  adjacent.
- `figsize=(14, 7)` — widen for many small intersections.
- `sort_categories_by="-input"` — preserves the `BACKENDS` order on the
  left. Use `"cardinality"` if you instead want the backend with the most
  fails on top.
- `show_counts=True` — annotate every bar with its count. Set `False` for a
  cleaner look in posters/slides.
- For a black-and-white print version, add `facecolor="black"` to the
  `UpSet(...)` constructor.

### 8. Reproducibility

- Pattern-set version: 883 live patterns from `OTPLibrary()` on branch
  `catalog/campaign-v10-2026-04`.
- Seed list: `tools/full_probe_9.py::_SEEDS` (six structural contexts
  covering NCHW / NLC / NC at two scales each).
- ONNX opset: 17, IR version: 8.
- RNG: `np.random.default_rng(0)`; pattern instantiation uses the pattern's
  own `instantiate(..., rng, 0)` signature so structural choices are
  deterministic per `(pattern, seed)` pair.
- Subprocess timeout: 90 s per pattern; CPU-only mask for TF/TFLite.
