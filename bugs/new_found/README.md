# XComp — New Bugs (1000-model campaign)

Five **fully self-contained, hand-built** Python reproducers for the
high-severity findings of the Xcomp 1000-model campaign (seed 2026-04-23).

Each script:
* builds a tiny ONNX graph in-memory with `onnx.helper`,
* runs it through onnxruntime (the reference) and one suspect backend
  via `trion.oracle.*Backend`,
* prints both outputs + diff metrics,
* exits **0 = bug reproduced**, 1 = ran cleanly, 2 = backend missing.

No witness files, no JSON metadata, no campaign output required.

## Coverage

| # | Script | Suspect backend | Bugs reproduced | Severity |
|---|--------|------------------|-----------------|----------|
| 1 | `rc_01_torchscript_topk_tile_layernorm.py` | torchscript | 75, 589, 920 | low (precision) |
| 2 | `rc_02_tvm_mul_zero_matmul_nan.py` | tvm | 667 | **HIGH** — TVM NaN |
| 3 | `rc_03_torch_compile_pad_edge_dwconv_chain.py` | torch_compile | 722 | **HIGH** — abs diff ~9 |
| 4 | `rc_04_xla_resize_linear_halfpixel.py` | xla (jax) | 91, 136 | **HIGH** — rel_L2 0.71 |
| 5 | `rc_06_tvm_simplify_expr_softmax_crash.py` | tvm | crashes 15, 170, 353, 355, 910 | **HIGH** — TVM crash |

In total the 5 scripts cover **7 divergence bugs + 5 compiler crashes** —
all of the high-severity findings whose root cause reduces to a small
hand-built ONNX graph.

> The remaining 13 lower-severity bugs (small-magnitude precision drift
> on matmul/LN-heavy graphs, OpenVINO segfaults that need the exact
> shape/structure of the campaign witness) do not reduce to a minimal
> hand-built graph and are not included here.

## How to run

From the **XComp repo root** (so that `trion/` is importable):

```bash
PYTHONPATH=. python bugs/new_found/rc_01_torchscript_topk_tile_layernorm.py
PYTHONPATH=. python bugs/new_found/rc_02_tvm_mul_zero_matmul_nan.py
PYTHONPATH=. python bugs/new_found/rc_03_torch_compile_pad_edge_dwconv_chain.py
PYTHONPATH=. python bugs/new_found/rc_04_xla_resize_linear_halfpixel.py
PYTHONPATH=. python bugs/new_found/rc_06_tvm_simplify_expr_softmax_crash.py
```

Or run them all and check the exit codes:

```bash
for f in bugs/new_found/rc_*.py; do
  echo "=== $(basename "$f") ==="
  PYTHONPATH=. python "$f" >/dev/null 2>&1 ; echo "exit=$?"
done
```

Expected:
```
=== rc_01_torchscript_topk_tile_layernorm.py ===   exit=0
=== rc_02_tvm_mul_zero_matmul_nan.py ===           exit=0
=== rc_03_torch_compile_pad_edge_dwconv_chain.py === exit=0
=== rc_04_xla_resize_linear_halfpixel.py ===       exit=0
=== rc_06_tvm_simplify_expr_softmax_crash.py ===   exit=0
```

## Per-script reference output

### rc_01 — TorchScript LayerNorm-of-constant precision  (bugs 75, 589, 920)

3 nodes: `TopK(k=1) → Tile → LayerNorm`. ORT keeps an fp32 residue
(~6e-4) for LN of a constant row; TS rounds it to exactly 0. Relative
oracle reports 100% divergence at the noise floor.

```
input row max  = 3.0660    (LN is applied to that constant tiled across D=256)
ORT (first 5)  = [-0.00063204 -0.00063204 ...]    ||ref||_inf = 6.320e-04
TS  (first 5)  = [0. 0. 0. 0. 0.]                  ||ts ||_inf = 0.000e+00
rel_L2(TS, ORT)       = 0.999999     <-- the fuzzer's metric
max_abs_diff(TS, ORT) = 6.320e-04    <-- the absolute disagreement
BUG REPRODUCED — kernel-level fp32 precision divergence ...
```

### rc_02 — TVM NaN on Mul(x, 0) → Add(scale) → MatMul  (bug 667)

3-node graph whose output must equal the constant `scale @ W`. ORT does;
TVM returns 128 NaNs.

```
ORT output (correct)   first 6: [ 0.7032 -0.3630 -0.5919  0.0751  0.2396  1.2486]
TVM output (buggy)     first 6: [nan nan nan nan nan nan]
TVM |out|_max = 0.0000e+00   NaN count = 128/128
BUG REPRODUCED — TVM returns NaN ...
```

### rc_03 — TorchInductor Pad(edge)+DepthwiseConv miscompile  (bug 722)

`Pad(edge) → DepthwiseConv(group=3,5×5) → BN → Relu → clipped affine →
1/√(x²+eps) → MatMul`. Interior matches ORT to fp32 precision; spatial
edges disagree by **9 in absolute value**.

```
ORT/TC first 4: [ 4.9539 -0.3506  2.0965 -0.0564] | [ 4.9539 -0.3506  2.0965 -0.0564]
rel_L2=0.090393  max_abs_diff=9.243e+00
  [3442] ref=  -5.816  tc=  +0.596  diff=-6.412
  [3449] ref=  -7.368  tc=  -0.467  diff=-6.902
  [3440] ref= -11.394  tc=  -2.151  diff=-9.243
BUG REPRODUCED — Inductor's Pad(edge)+dwconv fusion is unsound at the spatial border.
```

### rc_04 — XLA Resize(linear, half_pixel) antialiases on downsample  (bugs 91, 136)

Single `Resize` node. Upsample matches ORT to ~1e-5; **downsample**
diverges 10–70% because `jax.image.resize(method="linear")` applies an
implicit anti-alias prefilter that ONNX's spec does not.

```
Direction          rel_L2     max_abs   ORT[0]      XLA[0]
upsample 30→32      0.000003  1.037e-05    +0.1257    +0.1257
upsample 64→96      0.000004  2.021e-05    +0.1257    +0.1257
downsample 32→30    0.094058  3.856e-01    +0.1087    +0.0853 <-- BUG
downsample 32→15    0.709071  1.411e+00    +0.1257    +0.1832 <-- BUG
BUG REPRODUCED — XLA's Resize does not match ONNX spec when downsampling.
```

### rc_06 — TVM SimplifyExpr crash on `Sub(Add(m,m), m) → Softmax`  (5 crashes)

4-node graph `MatMul → Add(m, m) → Sub(_, m) → Softmax(axis=0)`. ORT
returns 64 ones (correct). TVM's algebraic simplifier rewrites
`Sub(Add(m,m), m)→m` but leaves a free variable; the relay module fails
the well-formed check.

```
ORT output (correct)   first 6 : [1. 1. 1. 1. 1. 1.]
ORT |out|_max = 1.0000e+00   sum = 64.0000
TVM crashed during compilation. Error tail:
  |   %0 = nn.dense(%model_input1, ...);
  |   %2 = nn.dense(%model_input,  ...);
  |   %3 = subtract(%1, %2) ...;
  |   nn.softmax(%3, axis=0) ...;
  | }
  | contains free variables: [Var(model_input, ty=TensorType([1, 64], float32))]
BUG REPRODUCED — TVM SimplifyExpr leaves a free variable.
```

## Prerequisites

| Script | Required backend |
|--------|------------------|
| rc_01  | `pip install onnxruntime onnx2torch torch` |
| rc_02  | Apache TVM (Python 3.10 worker — see `trion/oracle/tvm_backend.py`) |
| rc_03  | `pip install onnxruntime onnx2torch torch` |
| rc_04  | `pip install onnxruntime jax jaxlib` |
| rc_06  | Apache TVM |

If a required backend is missing the script exits with status `2`.

## Bug → reproducer cross-reference

```
bugs 75, 589, 920           →  rc_01
bug  667                    →  rc_02
bug  722                    →  rc_03
bugs 91, 136                →  rc_04
crashes 15, 170, 353, 355, 910 →  rc_06
```
