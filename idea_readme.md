# Trion — Complete Technical Documentation

This document gives a complete, self-contained explanation of how Trion works:
what patterns are, how models are generated, how inputs are mutated, how the oracle
scores a model, how the feedback loop learns, and what the experiments found.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Overview](#2-system-overview)
3. [Pattern Library](#3-pattern-library)
4. [Model Generation](#4-model-generation)
5. [Input Mutation](#5-input-mutation)
6. [Oracle — Scoring a Model](#6-oracle--scoring-a-model)
7. [Feedback — Credit Assignment and UCB Bandit](#7-feedback--credit-assignment-and-ucb-bandit)
8. [The Full Loop](#8-the-full-loop)
9. [Experimental Results](#9-experimental-results)
10. [Key Findings](#10-key-findings)

---

## 1. Problem Statement

Modern deep learning compilers — `torch.compile`, TVM, ONNXRuntime, TFLite,
OpenVINO, XLA, TorchScript — apply aggressive optimizations to neural network
graphs before executing them:

- **Operator fusion**: merge Conv + BN + ReLU into a single fused kernel
- **Constant folding**: pre-compute expressions with known values
- **Layout rewriting**: change tensor memory layout for better cache utilization
- **Dead code elimination**: remove operations whose results are never used
- **Quantization-aware passes**: lower precision for speed without changing semantics

These transformations are mathematically **supposed to be equivalent** to the
original unoptimized computation. But compilers are complex, and bugs exist.
When a compiler optimization is wrong, it silently changes the model's numerical
output — the inference call succeeds, no exception is raised, but the answer is
incorrect.

**Goal**: Automatically find neural network models where a DL compiler's
optimization passes introduce incorrect results or crashes.

**Challenge**: The space of neural network architectures is infinite. Naive random
graph generation produces mostly invalid or degenerate models. We need a smarter
approach.

---

## 2. System Overview

Trion is a **coverage-guided fuzzer** built around three ideas:

1. **Pattern-based model generation**: build models by composing human-designed
   ONNX subgraph patterns that are known to stress compiler code paths
2. **Differential oracle**: detect bugs by comparing outputs across backends and
   across optimization levels (optimized vs unoptimized)
3. **UCB bandit feedback**: learn which patterns are most effective at triggering
   bugs and sample them more often

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Trion Main Loop                              │
│                                                                      │
│   Pattern Library (124 patterns, 7 categories)                       │
│          │                                                           │
│          ▼                           ┌─────────────────────────┐    │
│   PatternAwareSearchSpace            │   CreditAssignment      │    │
│   (hierarchical sampler)  ◄──────── │   (UCB bandit)          │    │
│          │                update     └─────────────────────────┘    │
│          │ generate                          ▲                       │
│          ▼                                   │ reward signal         │
│   ONNX model  (K=6 patterns composed)        │                       │
│          │                                   │                       │
│          ▼                                   │                       │
│   InputMutator                               │                       │
│   (base + 3 mutations)                       │                       │
│          │                                   │                       │
│          ▼                                   │                       │
│   DiscrepancyOracle  ─── score S(m) ─────────┘                      │
│   ├─ pytorch_eager   (reference, no optimization)                    │
│   ├─ onnxruntime     (opt=True  AND  opt=False)                      │
│   ├─ torchscript     (opt=True  AND  opt=False)                      │
│   ├─ torch_compile   (opt=True  AND  opt=False)                      │
│   ├─ tvm             (opt=True  AND  opt=False)                      │
│   ├─ tensorflow      (opt=True  AND  opt=False)                      │
│   ├─ tflite          (opt=True  AND  opt=False)                      │
│   ├─ openvino        (opt=True  AND  opt=False)                      │
│   └─ xla             (opt=True  AND  opt=False)                      │
│          │                                                           │
│          ▼                                                           │
│   Save: bug_NNNN.onnx + _report.json  (if score ≥ threshold)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pattern Library

### What is a pattern?

A **pattern** (formally an OTP — Operator Test Pattern) is a hand-designed ONNX
subgraph that:

- Is **valid**: produces well-typed ONNX with correct shapes
- Is **composable**: accepts an input tensor and produces an output tensor of a
  predictable shape, so patterns can be chained
- **Stresses a specific compiler code path**: fusion, layout propagation,
  constant folding, etc.

Each pattern is implemented as a Python class with:
- `build(input_name, input_shape, initializers, nodes) → output_name, output_shape`
  — constructs the ONNX nodes and weights
- `is_compatible(ctx: StructuralContext) → bool`
  — checks whether the pattern can accept the current tensor shape (e.g., some
  patterns require 4D inputs, others work with any rank)

### The 7 Categories and 124 Patterns

Patterns are grouped into 7 categories based on **what compiler behavior they stress**:

---

#### Fusion (24 patterns)

These patterns encode the exact subgraphs that compilers try to fuse into single
kernels (Conv+BN+ReLU, MatMul+Bias+GELU, etc.). If the fusion is wrong, the
fused kernel produces a different answer than running each op separately.

```
conv_bn_relu            conv_bn_leakyrelu       conv_bn_sigmoid
conv_add_relu           matmul_bias_relu        matmul_bias_gelu
depthwise_conv_bn_relu  conv_relu_conv_bn       linear_layernorm_gelu
gap_linear              conv_bn_relu_maxpool    matmul_bias_sigmoid
conv_transpose_bn_relu  dilated_conv_bn_relu    conv_bn_silu
conv_bn_hardswish       conv_bn_relu6           grouped_conv_bn_relu
conv_asym_pad_bn        conv_bn_elu             pointwise_dw_block
conv_gelu               matmul_bias_tanh        avgpool_scale_bias
```

---

#### Layout (20 patterns)

These patterns apply reshape/transpose/permute chains that force the compiler to
track tensor layout across operations. Bugs appear when the compiler miscomputes
strides, transposes in the wrong order, or fails to propagate layout changes.

```
reshape_transpose_reshape   flatten_dense_unflatten     double_transpose
reshape_softmax_reshape     flatten_unsqueeze_concat    squeeze_unsqueeze
transpose_conv_nhwc         reshape_layernorm_reshape   channel_shuffle
pad_conv                    space_to_depth_block        depth_to_space_block
reflect_pad_conv            reshape_batched_matmul      unsqueeze_expand_mul
tile_conv                   slice_pad_concat            gather_reshape
dilated_max_pool            ceil_mode_avg_pool_conv
```

---

#### Broadcast (18 patterns)

Element-wise operations with non-trivial broadcasting semantics. Compilers
sometimes mishandle shape broadcasting when fusing or vectorizing element-wise ops.

```
expand_add_mul      reciprocal_mul      log_clamp
exp_div_softplus    sub_mul_add         mul_add_relu
sqrt_div_rms        l2_norm             hard_clamp_norm
swish               where_mask          cumsum
logsumexp_step      abs_neg_relu        euclidean_norm_broadcast
softmax_mul         clipped_affine      floor_ceil_round
```

---

#### Normalization (18 patterns)

Normalization subgraphs (LayerNorm, GroupNorm, RMSNorm, BatchNorm). Compilers
often special-case these with pattern-matched fused implementations. If the
pattern match is wrong or incomplete, the result diverges.

```
manual_layernorm        layernorm_relu          layernorm_residual_add
instancenorm_relu       rmsnorm                 groupnorm_relu
batchnorm_eval          stable_softmax          spatial_reduce_mean
layernorm_dropout_identity  manual_group_norm   l2_normalize
power_norm              ada_layer_norm          batchnorm_relu6
variance_whitening      instance_norm_1d        layernorm_temperature
```

---

#### Branch (16 patterns)

Multi-path graphs with residual connections, channel gating, and feature pyramid
structures. Compilers must correctly handle tensors that flow through multiple
branches and are merged. Shape mismatches and incorrect fusion across branch
boundaries are common bug sites.

```
split_transform_concat  residual_add_relu       conv_branch_add
dual_pool_add           se_block                glu
add_layernorm           concat_conv             multi_scale_conv_branch
aspp_dilated_branch     spatial_attention_cbam  fpn_branch
dense_residual_block    channel_gating_branch   max_avg_pool_fusion
se_residual
```

---

#### Constant (16 patterns)

Algebraically trivial subgraphs that an optimizer **should** fold or eliminate:
identity chains, zero-subtraction, divide-by-one, log-then-exp cancellation.
These are specifically designed to catch constant-folding bugs — cases where the
compiler "optimizes away" an op but computes the wrong constant in the process.

```
constant_add_mul        identity_chain          redundant_reshape
self_sub_zero           div_by_constant         pow_canonical
cast_roundtrip          where_const_cond        sqrt_reciprocal_mul
transpose_inverse_cancel  reshape_reshape_cancel  slice_full_range
pad_slice_noop          mul_by_reciprocal       log_exp_cancel
learned_temperature_scale
```

---

#### Attention (12 patterns)

Transformer attention variants. These stress compiler tiling, kernel fusion for
multi-head attention, and handling of mask operations. High-value targets because
compilers invest heavily in attention optimization.

```
scaled_dot_product_attention    multi_head_self_attention
causal_masked_attention         transformer_ffn
transformer_encoder_layer       gated_mlp_block
attention_with_bias             group_query_attention
kv_cache_attention              rotary_embedding_attention
self_attention_residual         multi_query_attention
```

---

## 4. Model Generation

### Step 1 — Seed a structural context

Each model starts from a seed tensor shape. Trion samples either a 4D image-like
tensor `[1, C, H, W]` or a 3D sequence tensor `[1, seq_len, features]`, with
randomized channel counts and spatial sizes from a fixed set:

```
C  ∈ {3, 16, 32, 64}
H=W ∈ {8, 16, 32}
seq_len ∈ {8, 16, 32}
```

The **StructuralContext** tracks the current tensor's shape and dtype as it
flows through the pattern composition chain.

### Step 2 — Hierarchical pattern sampling (K steps)

The model is built by composing **K=6** patterns. Each step:

1. **Sample a category** from the admissible categories:
   - "Admissible" means the category has at least one pattern compatible with
     the current context shape
   - Categories are sampled from a softmax distribution over UCB utility scores
     (see Section 7)

2. **Sample a pattern** within that category:
   - Only compatible patterns (those whose `is_compatible(ctx)` returns True)
     are eligible
   - Patterns are sampled from a softmax distribution over their individual
     UCB utility scores

3. **Instantiate the pattern**:
   - The pattern's `build()` method is called with the current output tensor name
     and shape
   - It returns ONNX nodes, weight initializers, and the new output name/shape
   - Weights are sampled randomly from `Normal(0, 0.1)` at each instantiation,
     making every model numerically unique

4. **Update context**:
   - The structural context is updated with the new output shape
   - The next pattern must be compatible with this new shape

### Step 3 — Assemble the ONNX model

After K patterns are composed, all collected nodes and initializers are assembled
into a single `onnx.ModelProto` with:
- One graph input (the seed tensor)
- One graph output (the final pattern's output)
- All intermediate tensors named uniquely per-model

The model is verified with `onnx.checker.check_model()` before being passed to
the oracle.

### Composition example

```
Seed shape: [1, 3, 32, 32]

Step 1: layout/dilated_max_pool        → [1, 3, 14, 14]
Step 2: branch/concat_conv             → [1, 64, 14, 14]
Step 3: broadcast/hard_clamp_norm      → [1, 64, 14, 14]
Step 4: fusion/depthwise_conv_bn_relu  → [1, 64, 14, 14]
Step 5: layout/reshape_batched_matmul  → [1, 64, 196]
Step 6: layout/gather_reshape          → [1, 64, 196]

Final ONNX: 6 subgraphs, ~40 ops, one input, one output
```

This is exactly `bug_0000.py` — a real bug with score 6.0 (max possible for 6 backends).

---

## 5. Input Mutation

Running each model on one fixed input is insufficient. Some compiler bugs only
manifest on specific activation distributions (sparse, boundary values, scaled).

For each generated model, Trion creates **1 base input + 3 mutations = 4 total inputs**.
The oracle scores all 4 and takes the maximum score.

### Base input

Sampled from `Normal(0, 1)` with the model's input shape and dtype.

### Mutation strategies (4 types, cycled)

| Strategy | What it does | Why |
|---|---|---|
| **Additive noise** | Add Gaussian noise at 1% of input std | Small perturbations sometimes expose numerical precision bugs |
| **Scale variation** | Multiply by 0.1×, 0.5×, 1.0×, or 2.0× | Tests range sensitivity of optimization |
| **Sparse activation** | Zero out 90% of values | Mimics sparse activations common in practice |
| **Boundary values** | Inject 0, ±1, ±0.5 at 5% of positions | Tests handling of exact values (divide-by-zero prevention, identity ops) |

**Deliberately excluded**: extreme values (×1000, ×1e-5, inf, NaN). These would
cause floating-point overflow/underflow that triggers false-positive discrepancies
unrelated to compiler optimization bugs.

---

## 6. Oracle — Scoring a Model

The oracle answers the question: **does this model expose a compiler bug?**

It runs the model on every available backend and computes a unified score.

### 6.1 Two Bug Detection Signals

#### Signal 1: Cross-backend divergence — S_diff

Compare each target backend's output against the **PyTorch eager reference**
(unoptimized, numerically exact execution):

```
S_diff(backend) = min(1.0,  ||y_target - y_ref||₂  /  max(||y_ref||₂ · δ,  10δ))
```

where `δ` is the tolerance (default 1% = 0.01).

- `y_ref` = PyTorch eager output (no compilation, no optimization)
- `y_target` = backend output with optimization enabled (`opt=True`)
- The denominator is tolerance-scaled so that a 1% relative difference gives score ≈ 1.0
- Capped at 1.0 per backend to prevent a single wild backend from dominating

If `S_diff(backend) > 0`, the backend's compiler output disagrees with the
correct PyTorch reference.

#### Signal 2: Optimization-induced deviation — Δ_opt

Compare the **same backend** running with and without optimization:

```
Δ_opt(backend) = min(1.0,  ||y_opt - y_noopt||₂  /  max(||y_noopt||₂ · δ,  10δ))
```

- `y_opt`   = backend output with optimization ON
- `y_noopt` = backend output with optimization OFF (baseline)
- If these differ, the optimizer changed the mathematical result

This is the most powerful signal: it does **not require knowing the correct
answer**. It purely asks: "does the compiler produce a different answer depending
on whether optimization is enabled?" If yes, the optimization is buggy.

#### Total score

```
S(m) = Σ_{backends}  [S_diff(backend) + Δ_opt(backend)]
```

Maximum possible score with 8 backends = 8×(1.0 + 1.0) = 16.0.

**Crashes are tracked separately and do NOT contribute to S(m).** A crash is a
binary event — it tells you the backend failed, but not by how much. Mixing
crash signals with numerical signals would make scores incomparable.

### 6.2 Crash Classification

Not every exception is a compiler bug. Some crashes come from the
ONNX→PyTorch conversion layer (`onnx2torch`), `torch.fx` tracing, or `dynamo`
graph capture — these are infrastructure limitations, not compiler optimization bugs.

Trion classifies each crash as either:

| Type | Meaning | Examples |
|---|---|---|
| `"frontend"` | Conversion/tracing failure — not a compiler bug | `onnx2torch`, `FakeTensor`, `make_fx`, `dynamo`, `symbolic_trace` |
| `"backend"` | Genuine compiler/runtime failure | `RuntimeError`, `INTERNAL ASSERT`, `shape mismatch`, `lax.`, `Not implemented` |

**Frontend crashes** are logged but do not affect scoring or pattern credit.
**Backend crashes** are saved as bugs (even with score=0) because they represent
the compiler failing on a valid model.

### 6.3 Oracle Execution Sequence

For each model + input:

```
1. Run pytorch_eager(model, input, opt=False)  → y_ref

2. For each target backend B:
   a. Run B(model, input, opt=True)   → y_opt
   b. Run B(model, input, opt=False)  → y_noopt
   
   c. Compute S_diff(B) from y_ref vs y_opt
   d. Compute Δ_opt(B) from y_opt vs y_noopt
   
   e. Record any crashes with their classification

3. total_score = Σ [S_diff(B) + Δ_opt(B)]
```

Every backend exception is caught independently — one backend crashing does
not prevent the others from running.

---

## 7. Feedback — Credit Assignment and UCB Bandit

After scoring a model, Trion updates its **per-pattern utility estimates** so
that future models use better patterns.

### 7.1 Credit Distribution

A model is composed of K=6 patterns. When the model scores S(m), the reward is
distributed **uniformly** to all K patterns that composed it:

```
r_t = S(m) / K        (per-pattern reward)
```

This is the simplest fair credit assignment: each pattern contributed equally to
the model, so each receives an equal share.

For each pattern p that appeared in the model:
```
R(p) ← R(p) + r_t
N(p) ← N(p) + 1
```

where `R(p)` is the cumulative reward and `N(p)` is the count of times pattern p
has been used.

The same update applies at the **category level** (e.g., the "fusion" category
receives the same per-pattern reward from any fusion pattern that was used).

### 7.2 Crash Handling in Credit Assignment

Three distinct cases:

| Situation | Credit update |
|---|---|
| Score > 0 (numerical divergence found) | Reward r_t = S(m)/K distributed to all K patterns |
| Backend crash, score = 0 | Small penalty: r_t = -0.05/K — gently de-prioritize patterns causing unproductive crashes |
| Frontend crash only | **No update** — the crash is an infrastructure issue, patterns are not responsible |

The asymmetry is intentional: patterns that cause backend crashes get a mild
penalty (they waste time without finding optimization bugs), but patterns involved
in frontend crashes are not penalized at all (it would be unfair to penalize them
for an onnx2torch conversion limitation).

### 7.3 UCB Utility Score

After updating rewards and counts, the utility of each pattern is computed as:

```
U(p) = R(p) / (N(p) + ε)  +  λ / √(N(p) + 1)
       └────────────────┘     └───────────────┘
         exploitation            exploration
         (average reward)        (UCB bonus for under-sampled patterns)
```

Parameters:
- `ε = 1e-8` — numerical stability to avoid division by zero
- `λ = 1.0` — exploration coefficient controlling the exploitation/exploration tradeoff
- `N(p) = 0` → large UCB bonus (unsampled patterns are tried early)
- `N(p) → ∞` → bonus shrinks, utility converges to average reward

The same formula is applied at the category level: `U(category)`.

### 7.4 Softmax Sampling

Patterns are not sampled by argmax (that would be greedy and miss rare bugs).
Instead, UCB utilities are converted to **softmax probabilities** with temperature T=2:

```
logit(p) = U(p)
scaled(p) = (logit(p) - max_logit) / T
prob(p) = exp(scaled(p)) / Σ exp(scaled(p'))
```

Temperature T=2 flattens the distribution relative to T=1, giving under-explored
patterns a meaningful chance to be sampled even when some patterns have
accumulated high rewards.

### 7.5 Hierarchical Sampling

Sampling is hierarchical — category first, then pattern within category:

```
1. Compute admissible categories (those with at least one compatible pattern)
2. Sample category g ~ Softmax(U(categories))
3. Compute admissible patterns within g (those compatible with current shape)
4. Sample pattern p ~ Softmax(U(patterns in g))
```

This ensures the search respects both **shape compatibility** (not all patterns
can accept all input shapes) and **learned utilities** at both levels.

---

## 8. The Full Loop

```python
for i in range(num_models):

    # 1. Generate model by composing K=6 patterns
    model = search_space.generate()          # hierarchical UCB sampling
    
    # 2. Generate inputs (base + 3 mutations)
    inputs = mutator.generate_all(model.input_shape)
    
    # 3. Score with oracle — take best score across all inputs
    best_report = max(
        [oracle.score(model, x) for x in inputs],
        key=lambda r: r.total_score
    )
    
    # 4. Update credit assignment
    if best_report.total_score > 0:
        credit.update(model.pattern_sequence, best_report.total_score)
    elif has_backend_crash:
        credit.update_crash(model.pattern_sequence, "backend")   # small penalty
    elif all_frontend_crash:
        pass                                                       # no update
    else:
        credit.update(model.pattern_sequence, 0.0)               # zero reward
    
    # 5. Update sampling policies with new utilities
    search_space.update_policies(
        credit.category_utilities(),
        credit.pattern_utilities()
    )
    
    # 6. Save bug if score above threshold
    if best_report.total_score >= bug_score_threshold:
        save(model.onnx_model, best_report)
    elif has_backend_crash:
        save_crash(model.onnx_model, best_report)
```

Over 1000 iterations, the UCB bandit **converges toward patterns that reliably
trigger compiler optimization bugs**, while still exploring new patterns via the
exploration term.

---

## 9. Experimental Results

### Campaign v6 — Configuration

| Parameter | Value |
|---|---|
| Models generated | 1,000 |
| Backends tested | 8 (onnxruntime, torchscript, torch_compile, tvm, xla, tensorflow, tflite, openvino) |
| Reference backend | pytorch_eager |
| Tolerance δ | 1% (0.01) |
| Patterns per model K | 6 |
| Mutations per model | 3 |
| Seed | 42 |
| Bug score threshold | 0.05 |

### Verified Bugs

Starting from 202 raw bugs, 7 were removed as duplicates (same normalized crash
signature already represented by a higher-scoring bug), leaving:

**195 verified unique bugs**

### Bug Type Breakdown

| Type | Count | % |
|---|---|---|
| Discrepancy-only | 184 | 94.4% |
| Crash-only | 7 | 3.6% |
| Crash + Discrepancy | 4 | 2.1% |

### Root Cause Classification

This is the most important result. For each bug, we ask: does the bug appear
**only when optimization is enabled**, or does it appear regardless?

| Root Cause | Count | Explanation |
|---|---|---|
| **Compiler optimization bug** | 188 (96%) | `opt=False` gives correct answer. `opt=True` gives wrong answer. The bug is introduced by the optimization pass. |
| **Compiler support gap** | 5 (2.6%) | Crashes with both `opt=True` and `opt=False`. The compiler cannot execute a valid ONNX operation regardless of optimization level. |
| **Invalid model edge case** | 2 (1.0%) | Even `pytorch_eager` crashes (`bug_0013`, `bug_0032`). The model generation produced an unsupported configuration. |

**96% of bugs are compiler optimization bugs** — they do not exist in the
unoptimized execution path. The optimizers introduce the errors.

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

`torch.compile` is the most affected (61% of all bugs). This is expected —
`torch.compile` applies the most aggressive optimizations: TorchDynamo graph
capture, Inductor backend, triton kernel generation, and operator fusion.

### Score Distribution

| Metric | Value |
|---|---|
| Minimum | 0.0505 |
| Maximum | 12.0000 |
| Mean | 1.6522 |

A score of 12.0 means 6 backends all had both S_diff=1.0 and Δ_opt=1.0 — total
breakdown across all tested compilers.

### 13 Unique Crash Categories

| Crash Category | Affected Backends |
|---|---|
| `padding_size_error` | torch_compile, torchscript, tvm, xla |
| `depth_to_space_unimplemented` | torch_compile, torchscript, tvm, xla |
| `shape_broadcast_mismatch` | tvm, xla |
| `grouped_conv_shape_error` | tvm, xla |
| `concat_shape_mismatch` | tvm, xla |
| `onnxruntime_shape_inference_error` | onnxruntime |
| `onnxruntime_slice_shape_error` | onnxruntime |
| `openvino_frontend_parse_error` | openvino |
| `tflite_maxpool_not_supported` | tflite |
| `tflite_conv_transpose_not_supported` | tflite |
| `tflite_extract_patches_not_supported` | tflite |
| `tensorflow_padding_type_error` | tensorflow, tflite |
| `xla_unsupported_op` | xla |

### Most Bug-Triggering Patterns

Ranked by number of bug appearances across 195 verified bugs:

| Rank | Pattern | Category | Bug appearances | Why it triggers bugs |
|---|---|---|---|---|
| 1 | `cast_roundtrip` | constant | 55 | Dtype cast chain that is mathematically identity; optimizers try to fold it and often compute the wrong constant |
| 2 | `squeeze_unsqueeze` | layout | 50 | Redundant shape ops; optimizers cancel them but sometimes leave incorrect shape tracking |
| 3 | `residual_add_relu` | branch | 27 | Residual connection; optimizer must correctly fuse the addition with surrounding ops |
| 4 | `add_layernorm` | branch | 27 | Add followed by LayerNorm; compilers often fuse this pattern and introduce errors |
| 5 | `floor_ceil_round` | broadcast | 23 | Integer-rounding ops; tricky for floating-point optimizers to handle correctly |
| 6 | `variance_whitening` | normalization | 21 | Manual variance computation; optimizer sometimes rearranges the division incorrectly |
| 7 | `div_by_constant` | constant | 20 | Division by constant; optimizers convert to multiply-by-reciprocal and may use wrong precision |
| 8 | `log_clamp` | broadcast | 19 | Log followed by clamp; fusion can introduce wrong clamp bounds |
| 9 | `sqrt_reciprocal_mul` | constant | 19 | `1/sqrt(x)` pattern; common in normalization, often incorrectly fused |
| 10 | `aspp_dilated_branch` | branch | 18 | Multi-scale dilated convolution branch; complex layout dependencies |

**Key insight**: The top-2 patterns (`cast_roundtrip`, `squeeze_unsqueeze`) are
algebraically trivial — they are supposed to be no-ops. The fact that they trigger
55 and 50 bugs respectively reveals a systemic weakness: **compilers are most
likely to introduce errors when optimizing operations they believe are trivial**.
The "it's just a reshape" or "it's just a cast" assumption leads to bugs.

---

## 10. Key Findings

### Finding 1: Compiler optimization is the dominant bug source

96% of bugs (188/195) only exist when optimization is enabled. The unoptimized
compiler correctly executes the same model. This confirms that the testing
methodology — running opt=True vs opt=False on the same backend — is both
necessary and sufficient for finding real optimizer bugs.

### Finding 2: torch.compile has the most bugs

119/195 bugs (61%) affect `torch.compile`. This reflects both the aggressiveness
of its optimization pipeline (TorchDynamo + Inductor + Triton) and its relative
immaturity compared to more established compilers like ONNXRuntime. It is also
the most widely deployed compiler in the PyTorch ecosystem, making these bugs
high-impact.

### Finding 3: Constant-folding patterns trigger the most bugs

The top-2 patterns by bug count are both from the `constant` category:
`cast_roundtrip` (55 bugs) and `squeeze_unsqueeze` (50 bugs). Both are
mathematically trivial (identity-like) operations that optimizers try to
eliminate. This suggests compilers have systematic weaknesses in their
constant-folding and dead-code-elimination passes.

### Finding 4: Discrepancy bugs outnumber crash bugs 28:1

184 numerical discrepancy bugs vs 7 crash bugs. Discrepancy bugs are much harder
to find manually because the program runs to completion with no error. This
confirms the value of the differential testing approach: without comparing
opt vs noopt outputs, these silent numerical errors would never be detected.

### Finding 5: UCB feedback improves pattern selection

The patterns sampled most frequently by the UCB bandit (`cast_roundtrip`,
`squeeze_unsqueeze`) are also the ones with the highest bug yield. This is
not a coincidence — the UCB bandit learned over 1000 iterations that these
patterns are high-value and sampled them more. Random sampling over 124 patterns
uniformly would have found fewer bugs per model.

---

## Artifacts

| File | Description |
|---|---|
| `bugs/` | 195 verified bugs: `.onnx` + `_report.json` per bug |
| `bugs/index.json` | Searchable index of all bugs (id, class, score, patterns, crash info) |
| `bugs/campaign_summary.json` | Full campaign statistics |
| `reproduce/bug_NNNN.py` | 195 self-contained reproduction scripts, one per bug |
| `reproduce_bugs.py` | Batch reproduction with filtering by compiler, type, ID |
| `trion_bugs_v4/` | Full 1000-model corpus (`--save-all` mode) |
| `trion/patterns/` | All 124 pattern implementations |
| `trion/oracle/` | All 9 backend wrappers + DiscrepancyOracle |
| `trion/feedback/` | UCB credit assignment |
| `trion/generation/` | PatternAwareSearchSpace + hierarchical sampler |
| `trion/mutation/` | InputMutator (4 strategies) |
| `run_trion.py` | CLI entry point |

---

## How to Reproduce a Specific Bug

Every bug has a standalone script:

```bash
python reproduce/bug_0035.py    # crash bug (onnxruntime)
python reproduce/bug_0060.py    # discrepancy bug (onnxruntime opt vs noopt)
python reproduce/bug_0000.py    # crash + discrepancy (multiple backends)
```

Each script:
1. Loads the ONNX model from `bugs/`
2. Runs the affected backend(s) with `opt=True` and `opt=False`
3. For crash bugs: confirms the exception still fires
4. For discrepancy bugs: computes `rel_diff = ||y_opt - y_noopt||₂ / ||y_noopt||₂`
   and confirms it exceeds 5%

No arguments needed. Install the relevant backend and run.
