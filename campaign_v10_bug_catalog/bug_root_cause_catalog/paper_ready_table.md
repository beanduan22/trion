# Paper-Ready Summary Table

110 unique validated cross-backend divergences from the 1,500-model campaign (campaign_v10), collapsed into 21 root-cause clusters.

Abbreviations: **OV** = OpenVINO, **TC** = torch.compile (Inductor), **TS** = TorchScript, **TVM** = Apache TVM, **TF** = TensorFlow, **XLA** = OpenXLA, **ORT** = ONNX Runtime (reference).

## Root causes

| RC | # Bugs | Affected Backend(s) | Trigger Pattern | Nature | Conf. | Repro |
|----|-------:|---------------------|-----------------|--------|-------|-------|
| RC-01 | 41 | TF, XLA | `MatMul → Transpose → Softmax → MatMul` (4-D attention) | backend bug | high | `repros/unique_0040_min.py` |
| RC-02 | 26 | TF, TC, XLA | `Conv → BN → Relu` with broadcast `Add`/`Mul` upstream | backend bug | high | `repros/unique_0064_min.py` |
| RC-03 | 5 | OV, TF, TC, XLA | `Cast(f32→f16) → Cast(f16→f32) → manual softmax` | uncertain | med | `repros/unique_0010_min.py` |
| RC-04 | 5 | TC | `Conv → Relu → Softmax → Expand → Add → Conv → Relu` | backend bug | med | `repros/unique_0034_min.py` |
| RC-05 | 5 | TF | `3× MatMul → Softmax(axis=0) → Greater/Where → Softmax` | backend bug | med | `repros/unique_0019_min.py` |
| RC-06 | 5 | OV, TF, TC, TS, TVM, XLA | `Add³ → Unsqueeze → Squeeze → bias+Softmax → LN` | spec ambiguity | med | `repros/unique_0095_min.py` |
| RC-07 | 4 | OV, TF, TC, TS, XLA | `TopK(k=1) → Tile → … → LN` (singleton axis dropped) | backend bug | high | `repros/unique_0074_min.py` |
| RC-08 | 3 | OV, TC, TS, TVM, XLA | depthwise-Conv + BN fold + Resize(nearest, asymmetric) | uncertain | med | `repros/unique_0088_min.py` |
| RC-09 | 3 | XLA | `MatMul(4-D) → Slice → Tile → MatMul → Resize` | backend bug | med | `repros/unique_0042_min.py` |
| RC-10 | 2 | TF, TC, TS, TVM, XLA | `Resize(nearest, asymmetric, round_prefer_floor)` | spec ambiguity | high | `repros/unique_0002_min.py` |
| RC-11a | 1 | OV, TS, TVM, XLA | `ReduceMean → Mul → Transpose → MatMul → Softmax → LN` | uncertain | low | `repros/unique_0004_min.py` |
| RC-11b | 1 | OV, TS, TVM | `Reciprocal(0) → Mul → Conv → BN → Relu` | spec ambiguity | low | `repros/unique_0007_min.py` |
| RC-11c | 1 | TF, TC | `Relu → Resize(linear, half_pixel, 5→4)` | backend bug | low | `repros/unique_0014_min.py` |
| RC-11d | 1 | TS, TVM, XLA | `Max → Min → LN → Residual-Add → Relu` | backend bug | low | `repros/unique_0035_min.py` |
| RC-11e | 1 | TF, TC, TS | `3× MatMul → Add(0) → Mul(1) → Greater → Where(const)` | spec ambiguity | low | `repros/unique_0050_min.py` |
| RC-11f | 1 | OV, TF, TC, TVM, XLA | `Add(resid) → Relu → MatMul → Softmax → MatMul → Conv` | backend bug | low | `repros/unique_0055_min.py` |
| RC-11g | 1 | TF, TS, XLA | `ReduceMean ∥ ReduceMax → Concat → Conv → Sigmoid → Mul` (CBAM) | backend bug | low | `repros/unique_0065_min.py` |
| RC-11h | 1 | TC, TVM | `Expand → Add → Mul → LN(axis=1)` | uncertain | low | `repros/unique_0073_min.py` |
| RC-11i | 1 | TS, XLA | `Exp → Sub(1) → MatMul → Tanh` (manual `expm1`) | backend bug | low | `repros/unique_0076_min.py` |
| RC-11j | 1 | OV, XLA | `Reciprocal(-0.0) → Relu` (signed-zero IEEE-754 corner) | spec ambiguity | low | `repros/unique_0107_min.py` |
| RC-11k | 1 | TS | `Mul → ReduceSum → Add(eps) → Sqrt → Div` (manual L2-norm) | backend bug | low | `repros/unique_0108_min.py` |

## Aggregated view

| | Count |
|-|-:|
| Unique validated bugs | 110 |
| Root-cause clusters | 21 |
| Likely backend bug clusters | 12 (**93 bugs**) |
| Likely spec-ambiguity clusters | 5 (**10 bugs**) |
| Uncertain / needs confirmation | 4 (**7 bugs**) |
| Pure crash bugs | 0 |
| ONNX conversion/export bugs (ORT ≠ pytorch_eager) | 0 |

All repros are ≤110 lines, build a valid ONNX model programmatically with tiny shapes, pass `onnx.checker`, and execute on ORT without the original campaign’s base64-encoded model bytes.
