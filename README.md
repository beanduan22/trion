# Trion — Deep-Learning Compiler Fuzzer

Trion is a coverage-guided differential fuzzer for deep-learning compilers. It
composes ONNX models from a library of optimization-trigger patterns (OTPs),
runs each model on multiple backends, and flags any cross-backend or
optimization-vs-no-optimization discrepancy as a candidate compiler bug.

The reference backend is **PyTorch eager**. Anything that diverges from it by
more than the tolerance δ (default `1e-3`) is treated as a discrepancy bug.
Backend crashes are tracked separately and classified as either *frontend*
(ONNX→PyTorch tracing artefacts; ignored) or *backend* (real compiler
failures; saved).

## How it works

```
TrionRunner
 │
 ├─ PatternAwareSearchSpace.generate()           # builds one ONNX model
 │   ├─ sample category   π_cat   (UCB-driven softmax)
 │   ├─ sample pattern    π_pat   (UCB-driven softmax)
 │   ├─ instantiate OTP   → ONNX subgraph
 │   └─ thread context    c_t → c_{t+1}
 │
 ├─ InputMutator.generate_all()                  # 1 base input + N mutations
 │
 ├─ DiscrepancyOracle.score()                    # reference vs targets
 │   ├─ run pytorch_eager (reference, no opt)
 │   ├─ for each target backend:
 │   │     run optimized + run unoptimized
 │   │     S_diff   = min(1, ‖tar − ref‖₂ / ‖ref‖₂ / δ)
 │   │     Δ_opt    = min(1, ‖tar⁺ − tar⁻‖₂ / ‖tar⁻‖₂ / δ)   (only if tar⁻ ≈ ref)
 │   └─ total_score = Σ (S_diff + Δ_opt)
 │
 ├─ CreditAssignment.update()                    # UCB reward back to patterns
 │   U(p) = R(p)/(N(p)+ε) + λ/√(N(p)+1)
 │
 └─ search_space.update_policies()               # softmax over z-scored utilities
```

A model is saved as a **bug** when:

- `total_score ≥ bug_score_threshold` (numerical discrepancy), or
- at least one target backend produced a real *backend*-classified crash.

Each saved bug becomes `bug_NNNNNN.onnx` plus a `bug_NNNNNN_report.json` with
the score breakdown, pattern sequence, input arrays, and the opt vs no-opt
output arrays for direct reproduction. Compiler crashes are written to a
sibling `compiler_crashes/` subdirectory.

## Pattern library

172 hand-written ONNX subgraph patterns across 7 semantic categories:

| Category      | Count | What it stresses                                              |
|---------------|------:|---------------------------------------------------------------|
| fusion        |    32 | Conv/BN/activation fusion, GEMM-bias-activation, depthwise    |
| layout        |    27 | Reshape/Transpose chains, pad/slice, space↔depth, dilation    |
| broadcast     |    26 | Element-wise ops with broadcast semantics                     |
| constant      |    25 | Algebraically trivial subgraphs the optimizer should fold     |
| normalization |    24 | LayerNorm, RMSNorm, GroupNorm, variance whitening             |
| branch        |    21 | Residuals, SE blocks, multi-path concat, FPN                  |
| attention     |    17 | SDPA, MHA, GQA, MQA, RoPE, KV-cache, FFN variants             |

Every pattern is fully deterministic at runtime. Weights are seeded by
`(out_c, in_c, k)` so different invocations of the same pattern with the same
shape produce identical initializers. There are **no random ONNX ops** in any
pattern (no `RandomNormal`, `RandomUniform`, `Multinomial`, `Bernoulli`, no
training-mode `Dropout`); the only `Dropout` nodes use `ratio=0` in inference
mode and reduce to identity.

## Backends

| Backend         | Class                  | Notes                                  |
|-----------------|------------------------|----------------------------------------|
| `pytorch_eager` | `PyTorchEagerBackend`  | Reference (always run unoptimized)     |
| `onnxruntime`   | `ONNXRuntimeBackend`   | Default target                         |
| `torchscript`   | `TorchScriptBackend`   | Default target                         |
| `torch_compile` | `TorchCompileBackend`  | Default target (Inductor)              |
| `tvm`           | `TVMBackend`           | LLVM target by default                 |
| `xla`           | `XLABackend`           | Via `tf2xla`                           |
| `tensorflow`    | `TFBackend`            | Via `onnx_tf`                          |
| `tflite`        | `TFLiteBackend`        | Via `onnx_tf` → TFLite converter       |
| `openvino`      | `OpenVINOBackend`      |                                        |
| `tensorrt`      | `TensorRTBackend`      | Optional, requires CUDA + TensorRT SDK |

Backends that fail to import are silently skipped at startup.

## Installation

```bash
pip install -r requirements.txt

# Optional backends — install only what you need
pip install tensorflow>=2.13 onnx-tf>=1.10   # TensorFlow / TFLite / XLA
pip install openvino>=2023.0                  # OpenVINO
# TVM:       build from source — https://tvm.apache.org/docs/install/
# TensorRT:  requires CUDA + TensorRT SDK
```

## Usage

```bash
# Default: 1000 models, backends = onnxruntime, torchscript, torch_compile
python run_trion.py

# All eight backends, 1000 models, δ = 1e-3
python run_trion.py \
  --num-models 1000 \
  --backends onnxruntime torchscript torch_compile tvm xla tensorflow tflite openvino \
  --tolerance 1e-3

# Save every generated model — useful for building a corpus
python run_trion.py --save-all --output-dir my_corpus

# Custom output directory and bug threshold
python run_trion.py --output-dir my_results --bug-threshold 0.1

# Quiet mode (suppress per-bug log lines)
python run_trion.py --quiet
```

### CLI flags

| Flag               | Default                                      | Description                                    |
|--------------------|----------------------------------------------|------------------------------------------------|
| `--num-models`     | 1000                                         | Total models to generate and test              |
| `--pattern-budget` | 6                                            | Max patterns composed per model                |
| `--backends`       | onnxruntime torchscript torch_compile        | Target backends (subset of supported list)     |
| `--output-dir`     | `trion_results`                              | Where to write `bug_NNNNNN.{onnx,_report.json}`|
| `--seed`           | 42                                           | RNG seed                                       |
| `--tolerance`      | 1e-3                                         | δ for discrepancy detection                    |
| `--bug-threshold`  | 0.05                                         | `total_score` cutoff to save as bug            |
| `--tvm-target`     | `llvm`                                       | TVM target string                              |
| `--tvm-opt-level`  | 3                                            | TVM optimization level                         |
| `--fp16`           | off                                          | Enable TensorRT fp16                           |
| `--save-all`       | off                                          | Save every model, not just bugs                |
| `--no-save`        | off                                          | Disable writing artefacts                      |
| `--quiet`          | off                                          | Suppress info-level logs                       |

Some `TrionConfig` fields are not exposed via CLI yet:
`exploration_coefficient` (UCB λ), `num_mutations_per_model`,
`max_model_bytes`, `default_channels`, `default_spatial_size`. Edit
`trion/config.py` directly if you need to change them.

## Output format

```
trion_results/
├── bug_000123.onnx              # minimal ONNX reproducer
├── bug_000123_report.json       # score, patterns, inputs, opt/noopt outputs
├── compiler_crashes/
│   ├── crash_000456.onnx        # model that crashed a backend
│   └── crash_000456_report.json # error message + classification
└── summary.json                 # campaign-wide statistics
```

`bug_NNNNNN_report.json` contains, among other things:

```json
{
  "model_id": 123,
  "total_score": 0.61,
  "n_valid_comparisons": 3,
  "s_diff":     {"torch_compile": 0.00, "tvm": 0.31, "onnxruntime": 0.00},
  "delta_opt":  {"torch_compile": 0.30, "tvm": 0.00, "onnxruntime": 0.00},
  "noopt_valid":{"torch_compile": true, "tvm": true,  "onnxruntime": true},
  "crashes": [],
  "crash_info": {},
  "errors": {},
  "pattern_sequence": [
    ["broadcast",     "reciprocal_mul"],
    ["normalization", "ada_layer_norm"],
    ["broadcast",     "mul_add_relu"],
    ["layout",        "slice_pad_concat"],
    ["constant",      "reshape_reshape_cancel"],
    ["branch",        "aspp_dilated_branch"]
  ],
  "input_shape": [1, 64, 32, 32],
  "bug_inputs":      {"model_input": [...]},
  "expected_outputs":{"torch_compile": [...]},
  "buggy_outputs":   {"torch_compile": [...]}
}
```

`expected_outputs[backend]` is the no-opt run, `buggy_outputs[backend]` is the
opt run, and the `pattern_sequence` is enough to deterministically rebuild the
graph from the library.

## Repository layout

```
trion/
├── run_trion.py                 # CLI entry point
├── requirements.txt
└── trion/
    ├── config.py                # TrionConfig dataclass
    ├── runner.py                # main loop (TrionRunner)
    ├── patterns/                # OTP library — 172 patterns
    │   ├── base.py              # OTP / PatternInstance
    │   ├── library.py           # OTPLibrary (registry)
    │   ├── fusion_patterns.py
    │   ├── layout_patterns.py
    │   ├── broadcast_patterns.py
    │   ├── normalization_patterns.py
    │   ├── branch_patterns.py
    │   ├── constant_patterns.py
    │   └── attention_patterns.py
    ├── generation/
    │   ├── context.py           # StructuralContext
    │   └── search_space.py      # PatternAwareSearchSpace + ONNX builder
    ├── mutation/
    │   └── input_mutator.py     # base + perturbation strategies
    ├── feedback/
    │   └── credit_assignment.py # UCB credit / policy update
    └── oracle/
        ├── oracle.py            # DiscrepancyOracle, OracleReport
        ├── base.py              # BackendBase / BackendResult
        ├── pytorch_backend.py
        ├── onnxruntime_backend.py
        ├── torchscript_backend.py
        ├── torch_compile_backend.py
        ├── tvm_backend.py
        ├── tf_backend.py
        ├── tflite_backend.py
        ├── openvino_backend.py
        ├── xla_backend.py
        └── tensorrt_backend.py
```

## Per-compiler coverage

Trion records **line and branch coverage** for every compiler under test on
every run. Python coverage uses `coverage.py` with the include filter
scoped to just that backend's installed package directory. C++ coverage
requires the backend to be rebuilt with `-fprofile-arcs -ftest-coverage`;
Trion's runner wraps itself in a parent/child pair so that after the inner
process exits (and libgcov flushes `.gcda` files), the outer process walks
the build tree, runs `gcov -i -b`, and merges the results into
`coverage_report.json`.

```
=== Per-Compiler Coverage ===
  backend                      py lines            py branches                  C++ lines               C++ branches
  onnxruntime           72/461 ( 15.6%)        30/140 ( 21.4%)     80562/1240672 (  6.5%)      34066/703272 (  4.8%)
```

Python coverage works out of the box with the stock pip wheels. For C++
coverage of a specific backend, see [docs/cpp_coverage.md](docs/cpp_coverage.md)
for per-compiler build recipes. Example with instrumented ORT:

```bash
# assuming ~/cov/ort_debug points at an ORT build-with-coverage
PYTHONPATH=~/cov/ort_debug LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    python run_trion.py --num-models 50 --backends onnxruntime
```

## Expanding the pattern library

The 172 built-in patterns give broad structural coverage but don't target
every known compiler-bug class. [`docs/pattern_candidates.md`](docs/pattern_candidates.md)
lists candidate new patterns mined from real bug reports in
`microsoft/onnxruntime`, `apache/tvm`, `pytorch/pytorch`, and
`openvinotoolkit/openvino`, each citing the source issue, minimal subgraph,
and root cause.

## Self-contained bug reproducers

For every bug saved under `output_dir/`, Trion writes
`bug_NNNNNN_repro.py` alongside the ONNX model. The reproducer inlines the
ONNX graph, the failing input, and the PyTorch-eager reference output as
base64 strings — no external files are needed to run it:

```bash
python trion_results/bug_000123_repro.py
# → exits 0 if the bug is reproduced, 1 otherwise
```

## License

MIT
