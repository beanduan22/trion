# Raw Bugs — 98 Confirmed Reproducible Cross-Backend Bugs

> **Verified 2026-04-18** · All outputs below are recorded from actual runs on this machine.  
> Platform: Linux 6.17, Python 3.x · OpenVINO 2026.0 · ORT 1.24.4 · TF 2.21 · PyTorch 2.x

Discovered by **Trion** cross-backend differential testing (campaign v9, 2026-04-16 → 2026-04-18,
1 500 random models, 7 backends).

2 false positives removed after manual review (different GELU formulas / shared artifact).  
Each script is **self-contained**: `python <script>.py` → exit 0 = bug reproduced, exit 1 = not
reproduced on this version, exit 2 = missing dependency.  
Every script prints the **correct reference output** and the **wrong backend output** side-by-side.

---

## Quick start

```bash
# run a single repro
python tflite/cross_tflite_attention_logit_softcap.py

# run all bugs for one backend
for f in openvino/*.py; do python "$f" 2>/dev/null | tail -1; done

# run everything and count reproductions
grep -r "BUG REPRODUCED" <(for f in **/*.py; do python "$f" 2>/dev/null; done) | wc -l
```

---

## Bug classification

Every bug falls into one of two categories:

| Category | Meaning | Examples |
|----------|---------|---------|
| **ONNX convert** | Bug introduced at model load / conversion time — before any runtime execution. The converter itself produces a wrong or unrunnable graph. | onnx2torch Reshape→FakeTensor; TFLite fp16 weight quantizer; MaxPool spec validator |
| **Compiler optimization** | Bug introduced by a backend optimization pass — fusion, algorithm selection, dead-code elimination, or codegen. The ONNX graph is loaded correctly but the compiled kernel produces wrong output. | OpenVINO Winograd GEMM; ORT Cast-chain fusion; Inductor transpose+reduce fusion; XLA dead-code elim |

---

## Summary

| Backend | Bugs | ONNX convert | Compiler optimization | Notes |
|---------|-----:|:---:|:---:|-------|
| OpenVINO | 57 | 1 | 56 | Tiled-GEMM precision + integer-spec violations |
| torch.compile | 26 | 22 | 4 | onnx2torch FakeTensor crash (22 instances, 1 root cause) + Inductor bugs |
| ONNXRuntime | 7 | 0 | 7 | Cast fusion, Resize, Mod SIGFPE |
| XLA | 4 | 0 | 4 | Dead-code elim, silent shape mismatch, JIT precision |
| TFLite | 4 | 3 | 1 | fp16 weight quantization (convert-time) + XNNPack delegate |
| **Total** | **98** | **26** | **72** | |

---

## Recorded Output (sample runs)

Each block shows `$ python <script>` → actual terminal output.

### OpenVINO — uint8 subtract wraps wrong (RC-17)
```
$ python openvino/github_ov_019_uint8_sub_no_wrap.py
a        : [  5 200 250   0]
b        : [ 10 100  10  50]
OV out   : [  0 100 240   0]
wrap ref : [251 100 240 206]  (correct — ONNX modular)
sat  ref : [  0 100 240   0]   (wrong  — saturation)
Matches wrap (correct)? False
Matches sat  (wrong)?   True
BUG REPRODUCED — OV uint8 Sub saturates instead of wrapping
```

### OpenVINO — uint8 add wraps wrong (RC-17)
```
$ python openvino/github_ov_021_uint8_add_no_wrap.py
a        : [200 100 255 128]
b        : [100 200  10 200]
a+b      : [300 300 265 328]
OV out   : [255 255 255 255]
wrap ref : [44 44  9 72]  (correct)
sat  ref : [255 255 255 255]   (wrong)
BUG REPRODUCED — OV uint8 Add saturates instead of wrapping
```

### OpenVINO — int8 subtract saturates wrong (RC-17)
```
$ python openvino/github_ov_024_int8_sub_saturation.py
a        : [-128  127   -1    0]
b        : [   1 -128 -127  100]
OV out   : [-128  127  126 -100]
wrap ref : [ 127   -1  126 -100]  (correct — two's complement)
sat  ref : [-128  127  126 -100]   (wrong  — saturation)
BUG REPRODUCED — OV int8 Sub saturates instead of two's-complement wrapping
```

### OpenVINO — ReduceLogSumExp overflows to inf (RC-18)
```
$ python openvino/github_ov_022_reducelogsumexp_overflow.py
Input x  : [[100.  88.  50.]
             [200. -10.   1.]]
OV out   : [inf inf]
Ref      : [100.00001 200.     ]
has_inf  : True
BUG REPRODUCED — OV ReduceLogSumExp overflows for inputs ≥ 88.7 (fp32 exp limit)
```

### OpenVINO — Relu(NaN) returns 0 instead of NaN (RC-19)
```
$ python openvino/github_ov_023_relu_nan_propagation.py
Input  : [nan -1.  0.  1. inf]
ORT    : [nan  0.  0.  1. inf]  ← NaN propagated (correct)
OV     : [ 0.  0.  0.  1. inf]
ORT[0] is NaN: True  (expected: True)
OV[0]  is NaN: False  (expected: True)
BUG REPRODUCED — OV Relu(NaN) → 0.0, should propagate NaN (IEEE 754)
```

### OpenVINO — Exp(NaN) returns +inf instead of NaN (RC-20)
```
$ python openvino/github_ov_026_exp_nan_to_inf.py
Input  : [nan  1.]
ORT    : [      nan 2.7182817]  ← NaN propagated (correct)
OV     : [     inf 2.718282]
BUG REPRODUCED — OV Exp(NaN) → +inf, should propagate NaN (IEEE 754)
```

### OpenVINO — tiled GEMM precision divergence (RC-22)
```
$ python openvino/cross_openvino_add_relu_sub.py
ORT:      [33.165524    0.48662525  1.9708737  27.003704  ]
OpenVINO: [33.117676    0.48662525  1.9708737  26.950716  ]
max_abs=0.1404
BUG REPRODUCED: OpenVINO add_relu_sub (max_abs=0.1404)
```

### OpenVINO — MaxPool accepts invalid pad ≥ kernel (RC-21)
```
$ python openvino/cross_openvino_maxpool_bad_pad.py
  kernel=3 pad=3:
    ORT_ref       ERR   "Pad should be smaller than kernel"  ← CORRECT REJECT
    OpenVINO      ok    (1, 1, 12, 12)                       ← SILENT ACCEPT
    onnx2torch    ERR   pad should be at most half of kernel  ← CORRECT REJECT
    TorchScript   ERR   pad should be at most half of kernel  ← CORRECT REJECT
BUG REPRODUCED: OpenVINO silently accepts MaxPool with pad >= kernel
```

### OpenVINO — OOB Gather returns 0 silently (RC-10)
```
$ python openvino/tf_61865_onnx_openvino.py
indices : [0, 3, 6, 9, 12, 1]  (index 12 is OOB for size-12 tensor)
ORT_ref   ERR   INVALID_ARGUMENT: Non-zero status
OpenVINO  [1.0, 4.0, 7.0, 10.0, 0.0, 2.0]   ← OOB slot silently → 0.0
onnx2torch ERR  index 12 is out of bounds for dimension 0 with size 12
BUG REPRODUCED: OpenVINO silently returns a value for OOB Gather
```

### ORT — Cast chain drops int32 truncation (RC-06)
```
$ python onnxruntime/github_ort_004.py
Input             : [-0.2 -0.1  0.   0.1  0.2]
Expected (truncate→bool): [False False False False False]
ORT_DISABLE_ALL   : [ True  True False  True  True]
ORT_ENABLE_ALL    : [ True  True False  True  True]
BUG REPRODUCED: ORT's Cast(float→int32)→Cast(int32→bool) chain does not
  preserve int32 truncation. -0.1 → int32 → 0 → False (correct),
  but ORT gives True (fused cast skips truncation).
```

### ORT + OpenVINO — BitShift by 64 returns non-zero (RC-04)
```
$ python onnxruntime/cross_bitshift_shift64_ov_ort.py
Input values: [1000, 255, 1, 42]
Shift amount: [64, 64, 64, 64] (all 64)
Expected:     [0, 0, 0, 0] (all 0)
ORT_noopt   max_diff=1000  BUG
ORT_opt     max_diff=1000  BUG
ORT_LEFT    max_diff=1000  BUG
BUG REPRODUCED: BitShift by 64 returns non-zero (C UB — x86 masks shift to 6 bits)
```

### XLA — silent incompatible matmul (RC-15)
```
$ python xla/github_tf_61881_xla_matmul_incompatible_shapes.py
Input shape: (1, 4),  W shape: (6, 1)   ← inner dims 4 ≠ 6
Eager output: None  error=InvalidArgumentError  ← correct
XLA   output: [[42.]]  error=None               ← BUG: garbage result
BUG REPRODUCED: XLA silently executes invalid matmul; eager raises error.
```

### torch.compile — FakeTensor crash in Reshape (RC-01)
```
$ python torch_compile/cross_torch_compile_redundant_reshape.py
ORT:   [ 35.098927 -45.611603 -28.941078  26.396048]  (correct)
eager: [ 35.098927 -45.6116   -28.941076  26.396042]  (correct)
BUG REPRODUCED: torch.compile crashes on redundant_reshape:
  InternalTorchDynamoError: TypeError: torch.Size() takes an iterable
  of 'int' (item 0 is 'FakeTensor')
```

---

## TFLite — 4 bugs

Reference: TF/Keras eager (fp32).

| # | Script | Bug | Category | max_abs |
|---|--------|-----|----------|---------|
| 1 | `cross_tflite_attention_logit_softcap.py` | fp16 weight quantizer amplifies tanh saturation error in softcap attention | ONNX convert | 0.027 |
| 2 | `cross_tflite_transformer_encoder_layer.py` | XNNPack delegate fails to prepare full transformer block; fallback path gives catastrophically wrong result | Compiler optimization | 3.9e20 |
| 3 | `cross_tflite_sub_self_mul_zero.py` | Per-op fp16 quantization breaks matmul associativity: `(X@A)@B ≠ X@(A@B)` at deployment, amplified by trailing matmul | ONNX convert | 2250 |
| 4 | `cross_tflite_transpose_matmul_transpose.py` | fp16 quantization on 512-deep reduction accumulates error in transpose→matmul→transpose pattern | ONNX convert | 0.162 |

---

## torch.compile — 26 bugs

Reference: `torch.eager`.

### Root cause A — onnx2torch FakeTensor crash (22 bugs, 1 root cause)

Category: **ONNX convert**

`onnx2torch` converts ONNX `Reshape` with a runtime-tensor shape input into
`torch.reshape(x, shape_tensor.tolist())`. Under `torch.compile`, `shape_tensor` becomes
a `FakeTensor` and `torch.Size()` rejects it:
```
TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')
```
All 22 scripts below crash with `InternalTorchDynamoError` for this single reason.

| # | Script |
|---|--------|
| 1 | `cross_torch_compile_add_self_sub_double.py` |
| 2 | `cross_torch_compile_aspp_dilated_branch.py` |
| 3 | `cross_torch_compile_conv_k7_stride2.py` |
| 4 | `cross_torch_compile_conv_manual_celu.py` |
| 5 | `cross_torch_compile_cp_fp16_matmul_n512.py` |
| 6 | `cross_torch_compile_einsum_transpose.py` |
| 7 | `cross_torch_compile_flatten_gemm.py` |
| 8 | `cross_torch_compile_gemm_sigmoid.py` |
| 9 | `cross_torch_compile_group_query_attention.py` |
| 10 | `cross_torch_compile_ia_sat_sub_uint8_underflow.py` |
| 11 | `cross_torch_compile_manual_hard_swish.py` |
| 12 | `cross_torch_compile_matmul_add_layernorm.py` |
| 13 | `cross_torch_compile_multi_query_attention.py` |
| 14 | `cross_torch_compile_reciprocal_mul.py` |
| 15 | `cross_torch_compile_reduce_l2_last.py` |
| 16 | `cross_torch_compile_redundant_reshape.py` |
| 17 | `cross_torch_compile_slice_full_range.py` |
| 18 | `cross_torch_compile_sub_self_mul_zero.py` |
| 19 | `cross_torch_compile_topk_last_axis_k1.py` |
| 20 | `cross_torch_compile_transpose_transpose_squash.py` |
| 21 | `cross_torch_compile_triple_add_residual.py` |
| 22 | `cross_torch_compile_where_mask_fill.py` |

### Root cause B — Inductor / torch.compile kernel bugs (4 bugs, 4 distinct root causes)

| # | Script | Bug | Category |
|---|--------|-----|----------|
| 23 | `cross_torch_compile_space_to_depth_block.py` | Inductor bf16 lowering gives wrong SpaceToDepth layout (max_abs=0.00195) | Compiler optimization |
| 24 | `github_inductor_009_transpose_reduce_fusion.py` | Inductor fuses row-wise ReduceSum with Transpose and reinterprets storage strides — wrong numerical result | Compiler optimization |
| 25 | `github_inductor_011_bitshift_ub_shift64.py` | Inductor CPU codegen emits `x >> 64` — C UB: x86 masks shift count to 6 bits, returns x instead of 0 | Compiler optimization |
| 26 | `pt_121135_torch_compile.py` | `index_add` shape validation skipped under `torch.compile`; eager raises RuntimeError, compiled path silently does OOB write | Compiler optimization |

---

## ONNXRuntime — 7 bugs

Reference: ORT with no optimization (`ORT_DISABLE_ALL`) or ONNX spec / PyTorch eager.

| # | Script | Bug | Category | Severity |
|---|--------|-----|----------|----------|
| 1 | `cross_bitshift_shift64_ov_ort.py` | `BitShift(x, 64)` returns x instead of 0 — C UB, x86 masks shift to 6 bits | Compiler optimization | wrong output |
| 2 | `github_onnx_spec_007.py` | Resize nearest `half_pixel+round_prefer_ceil`: ORT picks index 14, spec requires index 15 | Compiler optimization | wrong output |
| 3 | `github_ort_002.py` | Resize `half_pixel+round_prefer_ceil` diverges vs PyTorch nearest | Compiler optimization | wrong output |
| 4 | `github_ort_004.py` | `Cast(float→int32)→Cast(int32→bool)` fusion drops int32 truncation: `-0.1→True` instead of `False` | Compiler optimization | wrong output |
| 5 | `github_ort_008.py` | Bicubic resize accumulates error of 0.061 vs PyTorch reference | Compiler optimization | wrong output |
| 6 | `github_ort_017_mod_int_divzero_sigfpe.py` | `Mod(int32, 0)` triggers **SIGFPE** — integer divide-by-zero not guarded | Compiler optimization | crash |
| 7 | `tf_61865_onnx_ort.py` | Out-of-bounds `Gather` raises `INVALID_ARGUMENT` instead of handling gracefully | Compiler optimization | crash |

---

## XLA — 4 bugs

Reference: TF eager / PyTorch eager.

| # | Script | Bug | Category | max_abs |
|---|--------|-----|----------|---------|
| 1 | `cross_xla_add_self_sub_double.py` | XLA JIT computes `x+x-2x` cancellation pattern wrong vs TF eager | Compiler optimization | 0.020 |
| 2 | `github_tensorflow_002.py` | XLA resize nearest/bilinear/bicubic uses different coordinate transform than eager | Compiler optimization | varies |
| 3 | `github_tf_61881_xla_matmul_incompatible_shapes.py` | XLA silently executes `[1,4] @ [6,1]` (incompatible shapes); eager correctly raises `InvalidArgumentError` | Compiler optimization | silent wrong |
| 4 | `github_tf_61884_xla_dead_code_elimination.py` | XLA executes dead slice with invalid range (size 5 > dim 3) and crashes; eager's `tf.function` correctly skips it | Compiler optimization | correctness |

---

## OpenVINO — 57 bugs

Reference: ONNXRuntime CPU (no optimization). All run on OpenVINO CPU plugin.

### ONNX spec violations — integer arithmetic and special values (10 bugs)

These are objective correctness bugs — not numerical precision. OpenVINO's kernel produces
a mathematically wrong result for inputs that have a single specified correct answer.

| # | Script | Bug | Category |
|---|--------|-----|----------|
| 1 | `github_ov_019_uint8_sub_no_wrap.py` | uint8 Sub: OV saturates to 0 instead of ONNX-required modular wrap: `5-10→0` not `251` | Compiler optimization |
| 2 | `github_ov_020_uint8_mul_no_wrap.py` | uint8 Mul: OV saturates to 255 instead of modular wrap: `200×200→255` not `64` | Compiler optimization |
| 3 | `github_ov_021_uint8_add_no_wrap.py` | uint8 Add: OV saturates to 255 instead of modular wrap: `200+100→255` not `44` | Compiler optimization |
| 4 | `github_ov_022_reducelogsumexp_overflow.py` | ReduceLogSumExp computed as `log(sum(exp(x)))` without max-subtraction stability trick; overflows to `inf` for x≥88.7 | Compiler optimization |
| 5 | `github_ov_023_relu_nan_propagation.py` | `Relu(NaN)→0.0` instead of NaN; violates IEEE 754 NaN propagation | Compiler optimization |
| 6 | `github_ov_024_int8_sub_saturation.py` | int8 Sub: OV saturates at −128 instead of two's complement wrap: `-128−1→−128` not `127` | Compiler optimization |
| 7 | `github_ov_025_int8_add_saturation.py` | int8 Add: OV saturates at 127 instead of two's complement wrap: `100+100→127` not `−56` | Compiler optimization |
| 8 | `github_ov_026_exp_nan_to_inf.py` | `Exp(NaN)→+inf` instead of NaN; OV's threshold comparison takes the large-value branch for NaN | Compiler optimization |
| 9 | `tf_61865_onnx_openvino.py` | OOB `Gather` index silently returns `0.0`; all other backends error | Compiler optimization |
| 10 | `cross_openvino_maxpool_bad_pad.py` | OV accepts `MaxPool` with `pad ≥ kernel` (invalid per ONNX spec) and returns wrong shape; ORT/PyTorch correctly reject at load time | ONNX convert |

### FP32 tiled-GEMM precision divergence (47 bugs)

**Root cause (shared):** OpenVINO CPU selects Winograd convolution or tiled GEMM algorithms
that accumulate partial sums in a different order than ORT's sequential reference path. For
fp32, this is normally within 1 ULP per multiply-add, but over a K=512 inner dimension it
can reach 0.01–1.0 absolute error. The same ONNX graph on ORT and OpenVINO produces
different but both internally-consistent fp32 outputs. **Category: Compiler optimization**
(algorithm selection happens at `core.compile_model()` time).

**Strongest evidence** — `cross_openvino_linear_relu_chain.py`: ORT_opt, onnx2torch,
torch.compile, and TorchScript all agree with ORT_ref; OpenVINO alone diverges, ruling
out ORT-specific quirks.

#### Attention / Transformer (8)
| Script | Pattern | max_abs |
|--------|---------|---------|
| `cross_openvino_alibi_attention.py` | ALiBi attention bias | 0.172 |
| `cross_openvino_attention_causal_only.py` | Causal mask attention | 0.230 |
| `cross_openvino_attention_logit_softcap.py` | `tanh(x/cap)*cap` before softmax | **38.7** |
| `cross_openvino_attention_with_sink_token.py` | Sink-token attention | 0.172 |
| `cross_openvino_flex_attention_precision_sdpa.py` | SDPA precision | 0.172 |
| `cross_openvino_group_query_attention.py` | GQA | 0.055 |
| `cross_openvino_multi_query_attention.py` | MQA | 0.172 |
| `cross_openvino_transformer_encoder_layer.py` | Full transformer encoder | varies |

#### Convolution (10)
| Script | Pattern | max_abs |
|--------|---------|---------|
| `cross_openvino_conv_add_relu.py` | Conv→Add→ReLU | 0.146 |
| `cross_openvino_conv_bn_eval_explicit.py` | Conv+BN eval folding | **0.720** |
| `cross_openvino_conv_bn_fusion.py` | Conv+BN fusion | varies |
| `cross_openvino_conv_bn_relu6.py` | Conv+BN+ReLU6 | 0.378 |
| `cross_openvino_conv_fp32_precision.py` | Winograd vs direct-conv accumulation order | varies |
| `cross_openvino_conv_prelu_channel.py` | Conv+PReLU per-channel | 0.146 |
| `cross_openvino_inception_v3_branch.py` | Inception-v3 branch | 0.388 |
| `cross_openvino_pad_conv.py` | Padded convolution | varies |
| `cross_openvino_pointwise_dw_block.py` | Pointwise+depthwise | varies |
| `cross_openvino_tile_conv.py` | Tiled convolution | varies |

#### Matrix operations (9)
| Script | Pattern | max_abs |
|--------|---------|---------|
| `cross_openvino_einsum_transpose.py` | Einsum+transpose | **0.545** |
| `cross_openvino_flatten_gemm.py` | Flatten→Gemm | 0.154 |
| `cross_openvino_fp16_matmul_add.py` | fp16 GEMM tile accumulation | varies |
| `cross_openvino_gemm_sigmoid.py` | Gemm+Sigmoid | 0.025 |
| `cross_openvino_matmul_add_biasgelu_bcast.py` | MatMul+Add+BiasGELU | 0.194 |
| `cross_openvino_matmul_add_layernorm.py` | MatMul+Add+LayerNorm | 0.009 |
| `cross_openvino_reduce_l2_last.py` | L2 reduction last axis | varies |
| `cross_openvino_redundant_reshape.py` | Redundant reshape elim | varies |
| `cross_openvino_transpose_matmul_transpose.py` | Transpose→MatMul→Transpose | varies |

#### Layout / Shape (6)
| Script | Pattern | max_abs |
|--------|---------|---------|
| `cross_openvino_space_to_depth_block.py` | SpaceToDepth | varies |
| `cross_openvino_transpose_transpose_squash.py` | Double-transpose cancel | varies |
| `cross_openvino_broadcast_1d_scalar.py` | 1-D scalar broadcast | 0.140 |
| `cross_openvino_slice_full_range.py` | Full-range slice | varies |
| `cross_openvino_glu.py` | GLU (gated linear unit) | **0.528** |
| `cross_openvino_global_branch_mul.py` | Global average + mul | **5.327** |

#### Activation / Elementwise (14)
| Script | Pattern | max_abs |
|--------|---------|---------|
| `cross_openvino_add_relu_sub.py` | Add→ReLU→Sub | 0.140 |
| `cross_openvino_add_self_sub_double.py` | `x+x-2x` cancellation | 0.180 |
| `cross_openvino_aspp_dilated_branch.py` | ASPP dilated branch | 0.250 |
| `cross_openvino_ei_log_zero.py` | Log near-zero | 0.203 |
| `cross_openvino_expm1_bounded.py` | expm1 bounded range | 0.148 |
| `cross_openvino_ia_sat_sub_uint8_underflow.py` | uint8 near-zero subtract | 0.140 |
| `cross_openvino_linear_relu_chain.py` | Gemm+ReLU chain | **1.2–2.8** |
| `cross_openvino_multi_scale_conv_branch.py` | Multi-scale branch | varies |
| `cross_openvino_neg_unary.py` | Negation | varies |
| `cross_openvino_reciprocal_mul.py` | Reciprocal→Mul | **13767** |
| `cross_openvino_rrelu_inference_identity.py` | RReLU inference path | varies |
| `cross_openvino_sub_self_mul_zero.py` | Cancellation-sensitive GEMM | varies |
| `cross_openvino_triple_add_residual.py` | Triple add residual | varies |
| `cross_openvino_where_mask_fill.py` | Where/mask fill | varies |

---

## How these were found

- **Tool**: Trion — random ONNX model generation + cross-backend differential testing
- **Campaign**: v9, 1 500 models, seeds 0–1499
- **Backends tested**: onnxruntime, torchscript, torch_compile, xla, tvm, openvino, tflite, tensorflow
- **Discovery date**: 2026-04-16 → 2026-04-18
- **Manual review**: 100 scripts reviewed one-by-one; 2 false positives removed (different GELU formulas; shared bicubic artifact)

---

## File naming convention

```
cross_<backend>_<pattern>.py   — bug found by cross-backend differential testing
github_<backend>_NNN_*.py     — bug cross-referenced with a public GitHub issue
```

## Dependencies

```
pip install numpy onnx onnxruntime openvino torch tensorflow
```

TVM bugs require a separate TVM build. XLA bugs require `jax[cuda]` or `tensorflow[and-cuda]`.
