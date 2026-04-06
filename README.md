# Trion — Deep Learning Compiler Fuzzer

Trion is a **coverage-guided fuzzer for deep learning compilers**. It automatically generates ONNX models by composing operator patterns, runs them through multiple DL backends, and detects bugs when a backend crashes or produces numerically incorrect results compared to a PyTorch eager reference.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trion Pipeline                           │
│                                                                 │
│  Pattern Library (124 patterns, 7 categories)                   │
│         │                                                       │
│         ▼                                                       │
│  PatternAwareSearchSpace  ──UCB feedback──►  CreditAssignment   │
│         │                                         ▲            │
│         ▼                                         │            │
│  Generate ONNX model  (K=6 patterns composed)     │            │
│         │                                         │            │
│         ▼                                         │            │
│  InputMutator  (base + 3 mutations per model)     │            │
│         │                                         │            │
│         ▼                                         │            │
│  DiscrepancyOracle                                │            │
│    ├─ PyTorch eager  (reference)                  │            │
│    ├─ ONNXRuntime                                 │            │
│    ├─ TorchScript                                 │            │
│    ├─ torch.compile                               │            │
│    ├─ TVM                                         │            │
│    ├─ TensorFlow / TFLite                         │            │
│    ├─ OpenVINO                                    │            │
│    └─ XLA                                         │            │
│         │                                         │            │
│         ▼                                         │            │
│  Score (crash + rel_diff)  ──reward signal────────┘            │
│         │                                                       │
│         ▼                                                       │
│  Save bug: .onnx + _report.json                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | What it does |
|---|---|
| **Pattern Library** | 124 hand-crafted ONNX subgraph patterns across 7 semantic categories (fusion, layout, broadcast, normalization, branch, constant, attention). Each pattern is a composable building block. |
| **PatternAwareSearchSpace** | Selects K patterns per model using a UCB (Upper Confidence Bound) bandit. High-reward patterns are sampled more often over time. |
| **InputMutator** | Generates a base random input plus N mutations (shape/value perturbations) to increase oracle sensitivity. |
| **DiscrepancyOracle** | Runs the same ONNX model on all backends, compares every output to PyTorch eager reference. Reports relative L2 difference and whether a backend crashed. |
| **CreditAssignment** | Maintains per-pattern reward estimates. After each model, reward is distributed to the patterns that composed it. Frontend tracing failures (onnx2torch artefacts) are distinguished from real compiler crashes and handled separately to avoid unfair penalization. |

---

## Bug Detection Criteria

A **discrepancy bug** is flagged when:

```
rel_diff(backend_output, reference_output) > tolerance
```

The default tolerance via CLI is `1e-3` (0.1%). The `TrionConfig` dataclass default is `0.01` (1%) — always pass `--tolerance` explicitly to control this.

A **crash bug** is flagged when:
- The backend raises an exception or produces no output
- The crash is classified as `"backend"` type (real compiler failure, not a frontend tracing artefact from onnx2torch/dynamo)

Both types produce a `.onnx` model and a `_report.json`. Frontend-only failures (FakeTensor errors, dynamo graph breaks, onnx2torch conversion issues) are logged but **not** saved as bugs.

---

## Pattern Library — 124 Patterns

### Fusion (24)
Operator fusion patterns that compilers commonly try to collapse into fused kernels:
`conv_bn_relu`, `conv_bn_leakyrelu`, `conv_bn_sigmoid`, `conv_add_relu`, `matmul_bias_relu`, `matmul_bias_gelu`, `depthwise_conv_bn_relu`, `conv_relu_conv_bn`, `linear_layernorm_gelu`, `gap_linear`, `conv_bn_relu_maxpool`, `matmul_bias_sigmoid`, `conv_transpose_bn_relu`, `dilated_conv_bn_relu`, `conv_bn_silu`, `conv_bn_hardswish`, `conv_bn_relu6`, `grouped_conv_bn_relu`, `conv_asym_pad_bn`, `conv_bn_elu`, `pointwise_dw_block`, `conv_gelu`, `matmul_bias_tanh`, `avgpool_scale_bias`

### Layout (20)
Tensor shape transformation chains that expose layout-sensitive compiler paths:
`reshape_transpose_reshape`, `flatten_dense_unflatten`, `double_transpose`, `reshape_softmax_reshape`, `flatten_unsqueeze_concat`, `squeeze_unsqueeze`, `transpose_conv_nhwc`, `reshape_layernorm_reshape`, `channel_shuffle`, `pad_conv`, `space_to_depth_block`, `depth_to_space_block`, `reflect_pad_conv`, `reshape_batched_matmul`, `unsqueeze_expand_mul`, `tile_conv`, `slice_pad_concat`, `gather_reshape`, `dilated_max_pool`, `ceil_mode_avg_pool_conv`

### Broadcast (18)
Element-wise operations with broadcasting semantics:
`expand_add_mul`, `reciprocal_mul`, `log_clamp`, `exp_div_softplus`, `sub_mul_add`, `mul_add_relu`, `sqrt_div_rms`, `l2_norm`, `hard_clamp_norm`, `swish`, `where_mask`, `cumsum`, `logsumexp_step`, `abs_neg_relu`, `euclidean_norm_broadcast`, `softmax_mul`, `clipped_affine`, `floor_ceil_round`

### Normalization (18)
Normalization subgraphs that compilers often pattern-match and replace:
`manual_layernorm`, `layernorm_relu`, `layernorm_residual_add`, `instancenorm_relu`, `rmsnorm`, `groupnorm_relu`, `batchnorm_eval`, `stable_softmax`, `spatial_reduce_mean`, `layernorm_dropout_identity`, `manual_group_norm`, `l2_normalize`, `power_norm`, `ada_layer_norm`, `batchnorm_relu6`, `variance_whitening`, `instance_norm_1d`, `layernorm_temperature`

### Branch (16)
Multi-path subgraphs with residual connections and channel gating:
`split_transform_concat`, `residual_add_relu`, `conv_branch_add`, `dual_pool_add`, `se_block`, `glu`, `add_layernorm`, `concat_conv`, `multi_scale_conv_branch`, `aspp_dilated_branch`, `spatial_attention_cbam`, `fpn_branch`, `dense_residual_block`, `channel_gating_branch`, `max_avg_pool_fusion`, `se_residual`

### Constant (16)
Algebraically trivial subgraphs that optimizers should fold but sometimes mishandle:
`constant_add_mul`, `identity_chain`, `redundant_reshape`, `self_sub_zero`, `div_by_constant`, `pow_canonical`, `cast_roundtrip`, `where_const_cond`, `sqrt_reciprocal_mul`, `transpose_inverse_cancel`, `reshape_reshape_cancel`, `slice_full_range`, `pad_slice_noop`, `mul_by_reciprocal`, `log_exp_cancel`, `learned_temperature_scale`

### Attention (12)
Transformer attention variants that stress compiler fusion and tiling:
`scaled_dot_product_attention`, `multi_head_self_attention`, `causal_masked_attention`, `transformer_ffn`, `transformer_encoder_layer`, `gated_mlp_block`, `attention_with_bias`, `group_query_attention`, `kv_cache_attention`, `rotary_embedding_attention`, `self_attention_residual`, `multi_query_attention`

---

## Campaign Results — v6 (Verified)

**1000 models generated** | **8 backends tested** | **195 verified unique bugs**

Verification: 202 initial bugs → 7 duplicate crash bugs removed (same error signature already represented by a higher-scoring bug) → 0 false positives → **195 confirmed**.

### Bug Types

| Type | Count | Description |
|---|---|---|
| Discrepancy-only | 184 | Numerical output diverges from PyTorch eager reference |
| Crash-only | 7 | Backend crashes, no numerical output produced |
| Crash + Discrepancy | 4 | Both crash and numerical divergence observed |

### Bugs by Compiler

| Compiler | Total Bugs | Crashes | Discrepancies |
|---|---|---|---|
| torch.compile | 119 | 9 | 115 |
| TFLite | 54 | 16 | 46 |
| OpenVINO | 37 | 4 | 35 |
| ONNXRuntime | 21 | 6 | 18 |
| TVM | 15 | 12 | 9 |
| XLA | 15 | 12 | 9 |
| TensorFlow | 12 | 11 | 7 |
| TorchScript | 9 | 8 | 5 |

> Note: A single bug may affect multiple compilers; counts above reflect bugs where that compiler is the **primary** affected backend.

### Unique Crash Categories (13 types)

| Crash Category | Affected Backends | Representative Bug |
|---|---|---|
| `padding_size_error` | torch_compile, torchscript, tvm, xla | bug_0079 |
| `depth_to_space_unimplemented` | torch_compile, torchscript, tvm, xla | bug_0152 |
| `shape_broadcast_mismatch` | tvm, xla | bug_0023 |
| `grouped_conv_shape_error` | tvm, xla | bug_0009 |
| `concat_shape_mismatch` | tvm, xla | bug_0000 |
| `onnxruntime_shape_inference_error` | onnxruntime | bug_0013 |
| `onnxruntime_slice_shape_error` | onnxruntime | bug_0035 |
| `openvino_frontend_parse_error` | openvino | bug_0013 |
| `tflite_maxpool_not_supported` | tflite | bug_0017 |
| `tflite_conv_transpose_not_supported` | tflite | bug_0009 |
| `tflite_extract_patches_not_supported` | tflite | bug_0166 |
| `tensorflow_padding_type_error` | tensorflow, tflite | bug_0013 |
| `xla_unsupported_op` | xla (via tensorflow) | bug_0017 |

### Score Distribution

| Metric | Value |
|---|---|
| Min | 0.0505 |
| Max | 12.0000 |
| Mean | 1.6522 |

**Score formula:** `S(m) = S_diff(m) + Δ_opt(m)`
- `S_diff`: relative L2 distance between backend output and PyTorch eager reference, capped at 1.0 per backend
- `Δ_opt`: relative L2 distance between optimized and non-optimized run of the same backend, capped at 1.0
- Crashes are tracked separately and **do not inflate the score**

### Top Bug-Triggering Patterns

| Pattern | Category | Appearances in bugs |
|---|---|---|
| cast_roundtrip | constant | 55 |
| squeeze_unsqueeze | layout | 50 |
| residual_add_relu | branch | 27 |
| add_layernorm | branch | 27 |
| floor_ceil_round | broadcast | 23 |
| variance_whitening | normalization | 21 |
| div_by_constant | constant | 20 |
| log_clamp | broadcast | 19 |
| sqrt_reciprocal_mul | constant | 19 |
| aspp_dilated_branch | branch | 18 |

---

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# Optional backends (install only what you need)
pip install tensorflow>=2.13 onnx-tf>=1.10   # TensorFlow + TFLite + XLA
pip install openvino>=2023.0                  # OpenVINO
# TVM: build from source — https://tvm.apache.org/docs/install/
# TensorRT: requires CUDA + TensorRT SDK
```

---

## Usage

### Run the fuzzer

```bash
# Default: 1000 models, backends: onnxruntime + torchscript + torch_compile + xla + tvm
python run_trion.py

# All 8 backends, 1000 models, 0.1% tolerance
python run_trion.py \
  --num-models 1000 \
  --backends onnxruntime torchscript torch_compile tvm xla tensorflow tflite openvino \
  --tolerance 0.001

# Save every generated model (not just bugs) — useful for building a test corpus
python run_trion.py --save-all --output-dir my_corpus

# Custom output directory and score threshold
python run_trion.py --output-dir my_results --tolerance 0.001 --bug-threshold 0.05

# Quiet mode (suppress per-bug logs)
python run_trion.py --quiet
```

### Reproduce existing bugs

```bash
# List all confirmed bugs
python reproduce_bugs.py --list

# Reproduce all bugs
python reproduce_bugs.py

# Reproduce bugs for a specific compiler
python reproduce_bugs.py --class torch_compile
python reproduce_bugs.py --class tflite
python reproduce_bugs.py --class openvino

# Reproduce a single bug by ID
python reproduce_bugs.py --bug bug_0071

# Verbose: print full error trace on failure
python reproduce_bugs.py --verbose
```

---

## Bug Report Format

Each confirmed bug produces two files:

**`bugs/bug_NNNN.onnx`** — minimal ONNX model that triggers the bug

**`bugs/bug_NNNN_report.json`**:
```json
{
  "bug_id": "bug_0071",
  "sig": "12c0bd5f709b",
  "compiler_class": "torch_compile",
  "affected_compilers": ["torch_compile"],
  "total_score": 0.6123,
  "input_shape": [1, 64, 32, 32],
  "pattern_sequence": [
    ["broadcast", "reciprocal_mul"],
    ["normalization", "ada_layer_norm"],
    ["broadcast", "mul_add_relu"],
    ["layout", "slice_pad_concat"],
    ["constant", "reshape_reshape_cancel"],
    ["branch", "aspp_dilated_branch"]
  ],
  "delta_opt": {"torch_compile": 0.305},
  "crashes": [],
  "errors": {}
}
```

**`bugs/campaign_summary.json`** — full statistics including crash taxonomy, pattern usage, and per-compiler breakdown.

**`bugs/index.json`** — searchable index of all 195 verified bugs with `bug_id`, `class`, `compilers`, `score`, `patterns`, `has_crash`, and `crash_backends` fields.

---

## Repository Structure

```
trion/
├── run_trion.py              # CLI entry point
├── reproduce_bugs.py         # Bug reproduction and verification script
├── requirements.txt
├── bugs/                     # v6 campaign — 195 verified bugs
│   ├── index.json            # searchable index (195 entries)
│   ├── campaign_summary.json # full verified statistics + crash taxonomy
│   ├── bug_NNNN.onnx         # minimal ONNX reproducer (one per bug)
│   ├── bug_NNNN_report.json  # bug metadata (score, patterns, errors)
│   └── removed_duplicates/   # 7 duplicate crash bugs (kept for reference)
├── trion_bugs_v4/            # v4 corpus — all generated models (target: 1000)
└── trion/                    # core library
    ├── config.py             # TrionConfig dataclass
    ├── runner.py             # TrionRunner — main orchestration loop
    ├── patterns/             # Pattern library (124 patterns, 7 categories)
    │   ├── base.py           # OTP base class
    │   ├── library.py        # OTPLibrary — registry + UCB integration
    │   ├── attention_patterns.py
    │   ├── branch_patterns.py
    │   ├── broadcast_patterns.py
    │   ├── constant_patterns.py
    │   ├── fusion_patterns.py
    │   ├── layout_patterns.py
    │   └── normalization_patterns.py
    ├── generation/
    │   └── search_space.py   # PatternAwareSearchSpace (UCB model selection)
    ├── mutation/
    │   └── input_mutator.py  # InputMutator (shape + value perturbations)
    ├── oracle/
    │   ├── oracle.py         # DiscrepancyOracle + OracleReport
    │   ├── base.py           # BackendBase abstract class
    │   ├── pytorch_backend.py
    │   ├── onnxruntime_backend.py
    │   ├── torchscript_backend.py
    │   ├── torch_compile_backend.py
    │   ├── tvm_backend.py
    │   ├── tf_backend.py
    │   ├── tflite_backend.py
    │   ├── openvino_backend.py
    │   └── xla_backend.py
    └── feedback/
        └── credit_assignment.py  # UCB credit assignment
```

---

## Configuration Reference

All options live in `TrionConfig` (`trion/config.py`) and are exposed as CLI flags in `run_trion.py`.

| Parameter | CLI flag | CLI default | Config default | Description |
|---|---|---|---|---|
| `num_models` | `--num-models` | 1000 | 1000 | Total ONNX models to generate and test |
| `pattern_budget` | `--pattern-budget` | 6 | 6 | Patterns composed per model |
| `tolerance` | `--tolerance` | **1e-3 (0.1%)** | 0.01 (1%) | Relative L2 threshold for discrepancy detection |
| `bug_score_threshold` | `--bug-threshold` | 0.05 | 0.1 | Minimum score to save as a bug |
| `num_mutations_per_model` | _(not exposed)_ | — | 3 | Input mutations per model |
| `exploration_coefficient` | _(not exposed)_ | — | 1.0 | UCB λ — exploration vs exploitation |
| `seed` | `--seed` | 42 | 42 | RNG seed for reproducibility |
| `reference_backend` | _(not exposed)_ | — | pytorch_eager | Oracle reference backend |
| `target_backends` | `--backends` | onnxruntime, torchscript, torch_compile, xla, tvm | same | Backends under test |
| `save_all` | `--save-all` | False | False | Save every generated model, not just bugs |

> **Note:** The CLI `--tolerance` default (`1e-3`) differs from the `TrionConfig` default (`0.01`). The v6 campaign used `--tolerance 0.01`.

---

## License

MIT
