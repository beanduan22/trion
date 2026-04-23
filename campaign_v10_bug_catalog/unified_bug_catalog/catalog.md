# Unified Bug Catalog — Grouped by Root Cause

115 bug cases across 22 root-cause clusters.
Each cluster has one representative minimal repro in `repros/`.

Legend:
- **Nature**: `backend_bug` = single-target compiler/runtime bug;
  `crash_bug` = reproducible crash; `spec_ambiguity` = multiple backends
  implement a spec-level ambiguity differently; `uncertain` = single case
  or low-confidence, noted for completeness.
- **Affected**: backends whose output disagrees with the reference cluster (for divergence bugs), or the single crashing tool (for crashes).

---

## Backend / compiler bugs (single-target)

### RC-01 · XLA HLO lowering of attention-style matmul/transpose/softmax pipelines

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TF, XLA
- **# cases:** 41
- **Confidence:** high
- **Trigger pattern:** `MatMul -> Transpose -> Softmax -> MatMul (4-D batched attention)`
- **Member bug IDs:** unique_0000, unique_0001, unique_0003, unique_0012, unique_0016, unique_0018, … (+35 more)
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc01_min.py`](repros/rc01_min.py)

---

### RC-02 · XLA + Inductor conv/BN/Relu fusion under broadcast upstream

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TF, TC, XLA
- **# cases:** 26
- **Confidence:** high
- **Trigger pattern:** `Conv -> BatchNormalization -> Relu with broadcast Add/Mul upstream`
- **Member bug IDs:** unique_0006, unique_0008, unique_0009, unique_0011, unique_0015, unique_0017, … (+20 more)
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc02_min.py`](repros/rc02_min.py)

---

### RC-04 · Inductor-only Conv->Relu->Softmax chain miscompile

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TC
- **# cases:** 5
- **Confidence:** medium
- **Trigger pattern:** `Conv -> Relu -> Softmax -> Expand -> Add -> Conv -> Relu`
- **Member bug IDs:** unique_0013, unique_0034, unique_0053, unique_0057, unique_0096
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc04_min.py`](repros/rc04_min.py)

---

### RC-05 · TF graph-mode drift on triple-MatMul + Softmax(axis=0) + Greater/Where

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TF
- **# cases:** 5
- **Confidence:** medium
- **Trigger pattern:** `3x MatMul -> Softmax(axis=0) -> ReduceMean -> Greater/Where -> Softmax`
- **Member bug IDs:** unique_0019, unique_0045, unique_0071, unique_0077, unique_0085
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc05_min.py`](repros/rc05_min.py)

---

### RC-07 · TopK(k=1) singleton-axis dropped before LayerNorm

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** OV, TF, TC, TS, XLA
- **# cases:** 4
- **Confidence:** high
- **Trigger pattern:** `TopK(k=1, axis=-1) -> Tile -> ... -> LayerNormalization`
- **Member bug IDs:** unique_0074, unique_0081, unique_0093, unique_0100
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc07_min.py`](repros/rc07_min.py)

---

### RC-09 · XLA-only Resize/Squeeze/GlobalAveragePool lowering

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** XLA
- **# cases:** 3
- **Confidence:** medium
- **Trigger pattern:** `MatMul(4-D) -> Slice(step=2) -> Tile -> MatMul -> Resize(nearest, asymmetric)`
- **Member bug IDs:** unique_0027, unique_0032, unique_0042
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc09_min.py`](repros/rc09_min.py)

---

### RC-11c · TF + Inductor Resize(linear, half_pixel) 5->4

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TF, TC
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Relu -> Resize(linear, half_pixel, 5->4)`
- **Member bug IDs:** unique_0014
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11c_min.py`](repros/rc11c_min.py)

---

### RC-11d · branch Min/Max + LayerNorm + residual

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TS, TVM, XLA
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Max(x, c1) -> Min(., c2) -> LayerNorm -> Residual Add -> Relu`
- **Member bug IDs:** unique_0035
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11d_min.py`](repros/rc11d_min.py)

---

### RC-11f · residual-add-relu + 4D matmul; only TorchScript correct

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** OV, TF, TC, TVM, XLA
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Add(residual) -> Relu -> MatMul(4D) -> Softmax -> MatMul -> Conv`
- **Member bug IDs:** unique_0055
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11f_min.py`](repros/rc11f_min.py)

---

### RC-11g · CBAM spatial-attention ReduceMax axis ambiguity

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TF, TS, XLA
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `ReduceMean || ReduceMax -> Concat -> Conv -> Sigmoid -> Mul (CBAM spatial attention)`
- **Member bug IDs:** unique_0065
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11g_min.py`](repros/rc11g_min.py)

---

### RC-11i · manual expm1 + attention + resize layout

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TS, XLA
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Exp -> Sub(1) (manual expm1) -> MatMul -> Tanh`
- **Member bug IDs:** unique_0076
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11i_min.py`](repros/rc11i_min.py)

---

### RC-11k · TorchScript-only ReduceL2 + manual-LayerNorm

- **Nature:** backend_bug
- **Category:** numerical_divergence
- **Affected:** TS
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Mul(x,x) -> ReduceSum -> Add(eps) -> Sqrt -> Div (manual L2-norm)`
- **Member bug IDs:** unique_0108
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11k_min.py`](repros/rc11k_min.py)

---

## Reproducible crash bugs

### CR-01 · TVM relay.build 'contains free variables' on nn.dense(units=None) + add(y,y)

- **Nature:** crash_bug
- **Category:** compiler_crash
- **Affected:** TVM
- **# cases:** 5
- **Confidence:** high
- **Trigger pattern:** `relay.nn.dense(x, W, units=None) -> relay.add(y, y) inside relay.build`
- **Member bug IDs:** crash_000385__tvm_opt, crash_000905__tvm_opt, crash_000959__tvm_opt, crash_001225__tvm_opt, crash_001452__tvm_opt
- **Source catalog:** `crash_root_cause_catalog`
- **Minimal repro:** [`cr01_min.py`](repros/cr01_min.py)

---

## Specification ambiguities

### RC-06 · ONNX spec ambiguity in Squeeze/Unsqueeze + bias-softmax fusion

- **Nature:** spec_ambiguity
- **Category:** numerical_divergence
- **Affected:** OV, TF, TC, TS, TVM, XLA
- **# cases:** 5
- **Confidence:** medium
- **Trigger pattern:** `Add^3 -> Unsqueeze(axes=[2]) -> Squeeze(axes=[2]) -> bias+Softmax -> LayerNormalization`
- **Member bug IDs:** unique_0022, unique_0036, unique_0089, unique_0090, unique_0095
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc06_min.py`](repros/rc06_min.py)

---

### RC-10 · Resize nearest-asymmetric `round_prefer_floor` (spec ambiguity)

- **Nature:** spec_ambiguity
- **Category:** numerical_divergence
- **Affected:** TF, TC, TS, TVM, XLA
- **# cases:** 2
- **Confidence:** high
- **Trigger pattern:** `BN -> Mul/Div/Add -> Resize(nearest, asymmetric, round_prefer_floor, non-integer ratio)`
- **Member bug IDs:** unique_0002, unique_0051
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc10_min.py`](repros/rc10_min.py)

---

### RC-11b · Reciprocal(zero) + Conv + BN fuse corner

- **Nature:** spec_ambiguity
- **Category:** numerical_divergence
- **Affected:** OV, TS, TVM
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Reciprocal -> Mul -> Conv -> BN -> Relu (with zero in denominator branch)`
- **Member bug IDs:** unique_0007
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11b_min.py`](repros/rc11b_min.py)

---

### RC-11e · identity-folding removes Where(const) mask branch (spec ambiguity)

- **Nature:** spec_ambiguity
- **Category:** numerical_divergence
- **Affected:** TF, TC, TS
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `3x MatMul -> Add(0) -> Mul(1) -> Greater -> Where(const) (identity-folding breaks mask)`
- **Member bug IDs:** unique_0050
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11e_min.py`](repros/rc11e_min.py)

---

### RC-11j · Reciprocal(-0.0) sign (IEEE-754 corner)

- **Nature:** spec_ambiguity
- **Category:** numerical_divergence
- **Affected:** OV, XLA
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Reciprocal(-0.0) -> Relu (signed-zero IEEE-754 corner)`
- **Member bug IDs:** unique_0107
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11j_min.py`](repros/rc11j_min.py)

---

## Uncertain / low-confidence single cases

### RC-03 · fp16 cast-roundtrip + fused softmax-stability chain

- **Nature:** uncertain
- **Category:** numerical_divergence
- **Affected:** OV, TF, TC, XLA
- **# cases:** 5
- **Confidence:** medium
- **Trigger pattern:** `Cast(fp32->fp16) -> Cast(fp16->fp32) -> manual softmax (Sub(max)/Exp/ReduceSum/Div)`
- **Member bug IDs:** unique_0010, unique_0025, unique_0028, unique_0047, unique_0094
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc03_min.py`](repros/rc03_min.py)

---

### RC-08 · Depthwise Conv + BN fold AND Resize-nearest-asymmetric rounding (confounded)

- **Nature:** uncertain
- **Category:** numerical_divergence
- **Affected:** OV, TC, TS, TVM, XLA
- **# cases:** 3
- **Confidence:** medium
- **Trigger pattern:** `depthwise Conv(group=C) -> BN -> Resize(nearest, asymmetric, round_prefer_floor) -> Conv`
- **Member bug IDs:** unique_0005, unique_0023, unique_0088
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc08_min.py`](repros/rc08_min.py)

---

### RC-11a · row-reduce + mul + transpose + softmax + layernorm misfold

- **Nature:** uncertain
- **Category:** numerical_divergence
- **Affected:** OV, TS, TVM, XLA
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `ReduceMean -> Mul -> Transpose -> MatMul -> Softmax -> LayerNormalization`
- **Member bug IDs:** unique_0004
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11a_min.py`](repros/rc11a_min.py)

---

### RC-11h · Expand+Add+Mul with LayerNorm(axis=1) fold

- **Nature:** uncertain
- **Category:** numerical_divergence
- **Affected:** TC, TVM
- **# cases:** 1
- **Confidence:** low
- **Trigger pattern:** `Expand -> Add -> Mul -> LayerNormalization(axis=1)`
- **Member bug IDs:** unique_0073
- **Source catalog:** `bug_root_cause_catalog`
- **Minimal repro:** [`rc11h_min.py`](repros/rc11h_min.py)

---

