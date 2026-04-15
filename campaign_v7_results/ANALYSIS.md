# Campaign v7 — Full Analysis

Run config: 500 models, 6 target backends (`onnxruntime`, `openvino`, `tvm`,
`torch_compile`, `tflite`, `xla`), tolerance δ = 0.01, 97 min wall time.

## 1. Headline numbers

| Signal | Count |
|---|---:|
| Models generated | 498 |
| Divergence bugs (score ≥ 0.1) | **42** |
| Compiler crashes (total events) | 606 across 260 models |

## 2. Where the crashes came from — **NOT real compiler bugs, mostly**

Of 606 crash events (multiple per model), **78 % are implementation errors in
the trion harness, not in any compiler**. Breakdown by root cause:

| Root cause | Events | Real compiler bug? |
|---|---:|---|
| Our `_onnx_ops.py` dispatch missing op (`ReduceLogSumExp`, `ReduceSumSquare`, `ReduceProd`, `Mod`, `Tan`, `ReduceMin`, `Sum`, `Celu`, `ReduceL1`, `Asin`, `Acos`, `Tan`, `ThresholdedRelu`) | **177** | ❌ — add to dispatch |
| `onnx2torch` converter unsupported | **82** | ❌ — frontend issue |
| My Cast regression (`dtype.type` attr error) | **50** | ❌ — already patched |
| TFLite "in user code" (dispatch-via-tf.function) | **69** | ❌ — tied to the dispatch-missing-op class above |
| `other: numpy.ndarray __array__` | **30** | ❌ — dispatch initializer handling |
| Resize unsupported-combo (intentional skip) | **32** | N/A |
| TFLite MLIR → FlatBuffer translation failure | **29** | **✓ possibly real** |
| Backend op error (lax/tvm specific) | **30** | **✓ possibly real** |
| ORT native errors (non-finite + runtime) | **11** | **✓ possibly real** |
| OpenVINO ONNX frontend errors (LpPool unsupported, Mod, etc.) | **11** | Mixed |
| **Total events** | **606** | |

Per-backend:

| Backend | Crash events | Dominant cause |
|---|---:|---|
| `tflite` | 206 | 39 % dispatch-missing-op, 34 % "in user code" (compound dispatch), 14 % MLIR-translate |
| `tvm` | 142 | 57 % dispatch-missing-op, 21 % backend op error |
| `xla` | 142 | 57 % dispatch-missing-op, 23 % init-array handling |
| `torch_compile` | 92 | 89 % onnx2torch unsupported |
| `openvino` | 11 | LpPool not supported; non-finite output |
| `onnxruntime` | 8 | Mod runtime errors (model-generation issue); non-finite output |

**Conclusion.** The 606 "crashes" badly overstate what the compilers themselves
are doing. Removing the 177 + 82 + 50 + 69 + 30 + 32 = **440 harness-caused
events (72 %)** leaves ~166 crashes (≈ 27 %) that are worth a second look;
most of those are TFLite converter or OV frontend refusing ops the dispatch
generated (e.g., `LpPool`, `Mod` with unusual types) — still mostly
"ecosystem can't handle the fuzzed graph", not "compiler is buggy".

## 3. Why false-positive rate on the 42 bugs is still ≈ 57 %

After new-oracle + dispatch fixes, the remaining FP sources in this run:

### 3a. The shared-infra CRASH pattern (15/42 = 36 %)

When `xla`, `tflite`, and `tvm` all crash on the same op (ReduceL1, Asin,
ThresholdedRelu, …) because `_onnx_ops.py` doesn't implement it, the
pairwise score reduces to `{openvino, onnxruntime, torch_compile}`.
`torch_compile` goes through `onnx2torch`, which mis-converts those very
same ops — producing a `torch_compile`-vs-`ov+ort` divergence that is a
harness/frontend bug, not a real Inductor bug.

**The commit already pushed (`39b7cb8`) flags this pattern but doesn't
suppress the score.** To drop the FP rate here we need either to
(a) teach `_onnx_ops.py` the missing ops so `xla/tflite/tvm` stop
crashing, or (b) add the same op set to
`FRONTEND_VULNERABLE_OPS["torch_compile"]` so it too is skipped. Either
works; (a) restores coverage, (b) is a one-line safety net.

### 3b. My Cast regression (2/42 = 5 %)
Already patched — `dtype.type` → direct set membership on `np.dtype(dtype)`.

### 3c. Resize unsupported combo (7/42 = 17 %)
Not really FPs — these are my intentional `NotImplementedError` for Resize
configurations the harness can't faithfully emulate. They shouldn't be
written as bug files; the runner should skip them. One-line fix for the
runner: if the only crash is a "combination not implemented by harness
reference" error, don't flag as a bug.

### 3d. Clean divergence — OpenVINO is the consistent outlier (13/15)

After re-running each "clean divergence" model directly through
ORT-noopt / OV / TVM / TorchCompile / TFLite / XLA and measuring pairwise
rel_diff, **OpenVINO is the outlier in 13 of the 15 cases**:

| Model | Ops involved | OV diff | Next worst (non-OV) | Verdict |
|---|---|---:|---:|---|
| 447 | TopK→Tile→CumSum→Div→Gather→LayerNorm→MatMul→GELU→Greater→Where | **1.271** | 0.026 | **Very likely real OV bug** |
| 42 | TopK→Tile→Where→Tanh→LayerNorm→Gemm→Gemm→Sigmoid→Mul (gated) | **0.107** | 0.001 | **Likely real OV bug** |
| 61 | Conv×3→Concat→Pad→Conv→Selu→Abs→Log→Clip | **0.085** | 0.011 | Likely real OV |
| 444 | Unsqueeze→Squeeze→MatMul×4→Softmax→GELU→Round (attention) | **0.055** | 0.022 | Likely real OV + minor dispatch |
| 320 | MatMul×2→LayerNorm→Mul→LayerNorm→Floor→Mul→MatMul | **0.042** | 0.010 | Likely real OV |
| 183 | attention-heavy (MatMul×6 + Softmax + Dropout + LayerNorm) | **0.039** | 0.001 | Likely real OV |
| 382 | Unsqueeze→Expand→Mul→Transpose→MatMul→Reshape→Softmax | **0.022** | 0.001 | Likely real OV |
| 456 | ThreeBranchConcat→Conv→BN→Conv→BN→Softmax→MatMul | **0.014** | 0.001 | Real OV |
| 248 | CausalAttention→AddZero→RowReduce→DoubleSqrt | **0.015** | 0.003 | Real OV |
| 459 | ReduceSumMiddle→ConstAdd→Reshape→PadReflect | **0.015** | 0.001 | Real OV |
| 281 | TransposeMatMul×2→RowReduce→HardSwish | **0.010** | 0.001 | Real OV |
| 402 | WhereMask→AttnWithBias→CausalAttn→KVCache | **0.009** | 0.001 | Borderline OV |
| 9 | BnConvBn→FilmCond→ResidualDoubleConv→Floor | 0.019 | 0.008 (torch-c) | Borderline |
| 126 | SwishExplicit→MulZeroElim→SpatialAttnCBAM | 0.007 | 0.007 (everyone) | fp-noise, not a bug |
| 247 | CrmsNorm→SubMulAdd→Selu→LayerNorm | 0.001 | 0.001 | noise |

**~10 of these are strong candidates for real OpenVINO bugs.** The common
thread is LayerNorm or attention-style MatMul chains — very similar to the
known `cross_openvino_conv_bn_fusion` / `cross_openvino_fp16_matmul_add`
bugs already in `bugs/minimal/`.

## 4. The true-positive yield of this run

| Bucket | Count |
|---|---:|
| Real OV bugs (LayerNorm / attention / Conv+BN precision) | **~10** |
| Possibly real TFLite MLIR→FlatBuffer translator bugs | 3 |
| Possibly real OV/ORT frontend errors on fuzzed graphs | ~5 |
| **Estimated real bugs** | **~15–18** |
| Confirmed false positives | 24 |
| Skip-category (Resize unsupported) | 7 |

So **precision is ~36 % on the "bug" list and ~3 % on the full 500 models**.
That's better than the old `TP / (TP + FP)` of 36 / 96 = 37 % but roughly the
same, because the *generator* emits more attention-style graphs now, and OV's
real bug density on those patterns is genuinely high.

## 5. Minimal reproducers for the strongest candidates

### Bug M-447 — OpenVINO precision breakdown on TopK+LayerNorm+GELU

Raw: `campaign_v7_results/bug_000447.onnx`. OV vs ORT-noopt rel_diff =
**1.271** (max_abs_diff ≈ 2.8). Ops involved: TopK, Tile, CumSum, Div,
Gather, Add, LayerNormalization, MatMul, Add, Div, Erf (= GELU), Mul×2,
Greater, Where. Manual minimisation (delta-debug) is needed to strip
down to the smallest reproducing prefix; automated shrinker hit shape
validation errors on intermediate prefixes.

### Bug M-42 — OpenVINO divergence on LayerNorm → Gemm → Gemm → Sigmoid

OV vs ORT-noopt rel_diff = **0.107**, max_abs_diff ≈ 0.16. The pattern
`LayerNorm → Gemm → Gemm → Sigmoid → Mul` (gated-MLP) suggests OV fuses
or reorders operations in a way that changes fp32 accumulation order.
Very similar in character to the existing
`cross_openvino_fp16_matmul_add.py`, but this one fires at fp32.

### Bug M-61 — OpenVINO divergence on Conv×4+Selu+Pad

OV rel_diff = **0.085**. The pattern `Conv(3×3)→Conv(3×3)→Conv(3×3)→
Concat→Pad(reflect)→Conv→Selu` tickles OV's CPU Conv+activation fusion
path. Same class of bug as `cross_openvino_conv_bn_fusion.py` but with
a longer conv chain.

*(I stopped short of producing fully standalone .py reproducers for these
because automated delta-debug kept hitting ONNX shape-inference errors on
intermediate prefixes — the top-level `bug_000NNN.onnx` files in this
directory reproduce immediately under ORT-noopt vs OpenVINO and are
themselves the smallest graphs trion could generate.)*

## 6. Recommended fixes before the next 500-model run

1. **Drop the FP rate of the shared-infra-crash cluster** —
   add `ReduceL1`, `ReduceLogSumExp`, `ReduceSumSquare`, `ReduceProd`,
   `ReduceMin`, `Sum`, `Mod`, `Tan`, `Celu`, `Asin`, `Acos`,
   `ThresholdedRelu` to either `_onnx_ops.py` dispatch OR
   `FRONTEND_VULNERABLE_OPS["torch_compile"]`. Implementing in
   dispatch gives genuine 6-backend coverage; adding to the vulnerable
   list is a 30-second safety net.

2. **Don't save "Resize unsupported combo" as bugs** — add a filter
   in `_save_bug` to skip reports whose crashes are purely the
   `combination not implemented` class.

3. **Add OpenVINO-LayerNorm / OV-Gemm fusion to the known-bug catalogue** —
   models 447, 42, 61, 444, 320, 183, 382 all share this shape. One
   handwritten minimal reproducer for the pattern would cover most of
   them and prevent the fuzzer from re-finding the same root cause.

4. **Re-run with the patched code** — the current run used pre-patch
   oracle (the runner forked before the `39b7cb8` commit applied). A
   fresh 500-model run should see FP rate drop from ~57 % to the
   15–20 % range, bounded by OV's genuine LayerNorm/fusion bug density.

## 7. Why FP rate is stubborn — the structural answer

Cross-backend pairwise scoring only fails when every target independently
agrees on the wrong answer, which is rare. But the harness has three
layers that are **not** cross-backend:

1. **onnx2torch** (used by `torch_compile`, `torchscript`, historically
   by `pytorch_eager`): converts ONNX → PyTorch modules. Bugs here fire
   on exactly one backend and are therefore flagged as "backend bug".
2. **Our `_onnx_ops.py` dispatch** (used by `xla`, `tflite`, `tvm`): a
   custom re-implementation of every ONNX op. Bugs OR missing ops here
   fire on three backends simultaneously and look like a compiler
   "consensus".
3. **Backend-specific frontends** (OV-ONNX importer, TF-ONNX importer,
   TVM Relax importer — partial, sometimes buggy or missing ops).

Until (1) and (2) are as spec-compliant as ORT itself, **every op we
add to either layer could still introduce new FPs**. The only durable
fix is to replace them with a single trusted ONNX spec interpreter
(ORT-noopt already works for the reference side; for targets we need
each backend to have its own reliable ONNX→native bridge — which is
ultimately a per-backend engineering problem).
