# Root-Cause Catalog — 110 validated unique bugs

Every bug below is a numerical-divergence bug (cross-backend disagreement on a bit-for-bit-equivalent ONNX graph). The reference is `pytorch_eager` (onnx2torch + PyTorch CPU); ORT is always in the reference cluster.

No bug in this set is a pure crash. 28 bugs additionally trigger a side-crash inside an already-diverging backend's optimized build — these are recorded as co-symptoms, not separate entries.

---

## RC-01 · XLA HLO lowering of attention-style matmul/transpose/softmax pipelines

- **Category:** Compiler/backend bug
- **Affected backend(s):** tensorflow, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, torch_compile, tvm, openvino
- **Cases (41):** unique_0000, unique_0001, unique_0003, unique_0012, unique_0016, unique_0018, unique_0020, unique_0026, unique_0029, unique_0030, unique_0031, unique_0037, unique_0039, unique_0040, unique_0041, unique_0043, unique_0044, unique_0046, unique_0049, unique_0054, unique_0058, unique_0059, unique_0060, unique_0061, unique_0062, unique_0063, unique_0066, unique_0075, unique_0079, unique_0082, unique_0083, unique_0084, unique_0086, unique_0087, unique_0091, unique_0097, unique_0101, unique_0103, unique_0104, unique_0106, unique_0109
- **Trigger pattern:** `MatMul → Transpose → (Scale/Div) → Softmax → MatMul (4-D batched attention), often followed by LayerNormalization or Residual-Add.`
- **Symptom:** `tensorflow` and `xla` agree with each other, the other five backends agree with pytorch_eager; worst rel_l2 ≈ 0.05 – 1.0.
- **Why this is the likely root cause:** Both TF and XLA lower ONNX through TF-lite/TF-XLA path and finally into XLA HLO. XLA's HLO fusion pass rewrites `transpose + batched-matmul + softmax + matmul` into a fused kernel that differs from the 'transpose-then-matmul-then-softmax' semantics the ONNX spec (and all other backends) implements. 41/110 bugs share this shape, which is a strong cluster-level signal.
- **Confidence:** high
- **Representative minimal repro:** `repros/unique_0040_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0000.py … unique_0109.py}`

---

## RC-02 · XLA + Inductor conv/BN/Relu fusion

- **Category:** Compiler/backend bug
- **Affected backend(s):** tensorflow, torch_compile, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, tvm, openvino
- **Cases (26):** unique_0006, unique_0008, unique_0009, unique_0011, unique_0015, unique_0017, unique_0021, unique_0024, unique_0033, unique_0038, unique_0048, unique_0052, unique_0056, unique_0064, unique_0067, unique_0068, unique_0069, unique_0070, unique_0072, unique_0078, unique_0080, unique_0092, unique_0098, unique_0099, unique_0102, unique_0105
- **Trigger pattern:** `Conv → BatchNormalization → (Add|Mul) → Relu, frequently with a residual-branch Add or a 1×1×1×1 broadcast constant in the fused chain.`
- **Symptom:** The three 'aggressive-optimizer' backends (TF/XLA + torch.compile Inductor) agree on a wrong answer; the four 'spec-preserving' backends agree with pytorch_eager. rel_l2 ≈ 0.05 – 1.0.
- **Why this is the likely root cause:** Inductor, TF-XLA, and XLA all fold BatchNorm into the preceding Conv weights and biases before execution. When an additional element-wise op (broadcast Mul or residual Add) sits between BN and Conv, the folding changes the order of operations — mathematically equivalent in fp64, but not in fp32/bf16. 26 bugs share this exact Conv+BN+Relu shape.
- **Confidence:** high
- **Representative minimal repro:** `repros/unique_0064_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0006.py … unique_0105.py}`

---

## RC-03 · fp16 cast-roundtrip + fused softmax-stability chain

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, tensorflow, torch_compile, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, tvm
- **Cases (5):** unique_0010, unique_0025, unique_0028, unique_0047, unique_0094
- **Trigger pattern:** `Cast(fp32→fp16) → Cast(fp16→fp32) → manual softmax (Sub(max) → Exp → ReduceSum → Div), inside a larger model with LayerNorm/Add residuals.`
- **Symptom:** The four optimizer-heavy backends (OV + TF + torch.compile + XLA) diverge as a cluster. TVM+ORT+TS preserve the cast roundtrip. rel_l2 0.06 – 1.0.
- **Why this is the likely root cause:** Pattern `cp_roundtrip_fp16_small` introduces a round-trip cast that looks like dead code to aggressive folders; OV, TF, XLA, and Inductor all eliminate it, gaining fp32 precision. TVM, ORT, and TorchScript preserve the cast and its precision loss. The 'correct' answer per ONNX spec is the precision-lossy one.
- **Confidence:** medium
- **Representative minimal repro:** `repros/unique_0010_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0010.py … unique_0094.py}`

---

## RC-04 · Inductor-only Conv→Relu→Softmax chain miscompile

- **Category:** Compiler/backend bug
- **Affected backend(s):** torch_compile
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, tvm, openvino, tensorflow, xla
- **Cases (5):** unique_0013, unique_0034, unique_0053, unique_0057, unique_0096
- **Trigger pattern:** `Conv → Relu → Softmax, usually followed by Expand + Add into another Conv→Relu block.`
- **Symptom:** Only torch.compile diverges; everyone else agrees. rel_l2 ≈ 0.06 – 1.0.
- **Why this is the likely root cause:** Inductor applies a 'Conv + Relu + Softmax' vertical fusion (Triton kernel) when it sees this exact shape. The fused kernel loses Softmax's row-wise normalization because the partial-sum is computed per-tile rather than per-row.
- **Confidence:** medium
- **Representative minimal repro:** `repros/unique_0034_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0013.py … unique_0096.py}`

---

## RC-05 · TensorFlow graph-mode numerical drift on triple-MatMul + softmax-axis-0 attention

- **Category:** Compiler/backend bug
- **Affected backend(s):** tensorflow
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, torch_compile, tvm, openvino, xla
- **Cases (5):** unique_0019, unique_0045, unique_0071, unique_0077, unique_0085
- **Trigger pattern:** `MatMul → MatMul → MatMul → Softmax(axis=0) → Add(residual) → Mean → Greater/Where → Softmax → MatMul.`
- **Symptom:** Only TF (eager graph) diverges; XLA matches the reference (so it is *not* XLA fusion). rel_l2 ≈ 0.07 – 0.67.
- **Why this is the likely root cause:** When the Softmax axis is 0 (non-default) and sits inside a Greater/Where mask pattern, TF's graph executor reorders the reduction for a gather-like fused kernel; XLA does not. The divergence is between the TF runtime and TF-XLA, not between XLA and the world.
- **Confidence:** medium
- **Representative minimal repro:** `repros/unique_0019_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0019.py … unique_0085.py}`

---

## RC-06 · ONNX-spec ambiguity in Squeeze/Unsqueeze + bias-softmax fusion

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, tensorflow, torch_compile, torchscript, tvm, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime
- **Cases (5):** unique_0022, unique_0036, unique_0089, unique_0090, unique_0095
- **Trigger pattern:** `Add → Add → Add (triple residual) → Unsqueeze/Squeeze reshape → bias-Softmax(axis) → LayerNormalization.`
- **Symptom:** Only ORT + pytorch_eager match; every other backend gives its own distinct answer. rel_l2 0.13 – 1.0.
- **Why this is the likely root cause:** Squeeze/Unsqueeze with an explicit axes attribute was moved from attribute to input between ONNX opset 11 and 13; different compiler frontends interpret the axis list differently when combined with a downstream bias+Softmax fusion. ORT is the reference implementation and onnx2torch mirrors its behaviour, so pytorch_eager and ORT form a 2-way island.
- **Confidence:** medium
- **Representative minimal repro:** `repros/unique_0095_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0022.py … unique_0095.py}`

---

## RC-07 · TopK(k=1) result-shape handling feeding LayerNorm

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, tensorflow, torch_compile, torchscript, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, tvm
- **Cases (4):** unique_0074, unique_0081, unique_0093, unique_0100
- **Trigger pattern:** `TopK(k=1, axis=last) → Tile → Add/Mul → Concat → Slice → LayerNormalization.`
- **Symptom:** Five backends return totally wrong numbers (rel_l2 = 1.000 on every bug); only TVM + ORT preserve the TopK rank/value split correctly.
- **Why this is the likely root cause:** TopK with k=1 on the last axis produces a singleton dimension whose squeeze/unsqueeze handling diverges across compilers. When the value stream is then fed into LayerNorm whose axis is inferred from rank, five compilers pick the wrong axis.
- **Confidence:** high
- **Representative minimal repro:** `repros/unique_0074_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0074.py … unique_0100.py}`

---

## RC-08 · Depthwise Conv + BN fold and Resize-nearest-asymmetric rounding

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, torch_compile, torchscript, tvm, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, tensorflow
- **Cases (3):** unique_0005, unique_0023, unique_0088
- **Trigger pattern:** `Conv(group=C) → BatchNorm → Relu → Resize(mode=nearest, coordinate_transformation=asymmetric) → Conv → Add → Mul → Add → LayerNormalization.`
- **Symptom:** Five backends cluster on a wrong answer; only TF + ORT + pytorch_eager are correct. rel_l2 0.06 – 1.0.
- **Why this is the likely root cause:** Two co-occurring issues: (1) depthwise-conv + BN folding yields different numerics in compilers that fold vs. compilers that do not; (2) Resize nearest-asymmetric with round_prefer_floor is spec-ambiguous — TF and ORT pick the floor, others pick nearest-even.
- **Confidence:** medium
- **Representative minimal repro:** `repros/unique_0088_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0005.py … unique_0088.py}`

---

## RC-09 · XLA-only Resize/Squeeze/GlobalAveragePool lowering

- **Category:** Compiler/backend bug
- **Affected backend(s):** xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, torch_compile, tvm, openvino, tensorflow
- **Cases (3):** unique_0027, unique_0032, unique_0042
- **Trigger pattern:** `MatMul(4-D batch) → Slice(step=2) → Tile → MatMul → Mul → Add → Relu → Resize, or GlobalAveragePool → Flatten → Unsqueeze/Squeeze.`
- **Symptom:** Only XLA diverges, always with rel_l2 = 1.0 (total mismatch); TF via XLA does NOT diverge (handled correctly at TF level).
- **Why this is the likely root cause:** XLA's standalone (non-TF) frontend handles the Resize + 4-D batched matmul shape chain with a different layout than TF-XLA does. The TF-XLA bridge rewrites the model before HLO; standalone XLA does not.
- **Confidence:** medium
- **Representative minimal repro:** `repros/unique_0042_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0027.py … unique_0042.py}`

---

## RC-10 · Resize nearest-asymmetric with `round_prefer_floor` semantics

- **Category:** Compiler/backend bug
- **Affected backend(s):** tensorflow, torch_compile, torchscript, tvm, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, openvino
- **Cases (2):** unique_0002, unique_0051
- **Trigger pattern:** `BatchNorm → Mul/Div/Add → Resize(mode=nearest, coordinate_transformation=asymmetric, nearest_mode=round_prefer_floor) → Mul → Greater/Where → Tanh → Add → Mul.`
- **Symptom:** Five backends cluster on a totally different upsample result (rel_l2 = 1.0); only OV + ORT follow the ONNX spec's `round_prefer_floor` rule.
- **Why this is the likely root cause:** ONNX nearest-mode has four variants (round_prefer_floor, round_prefer_ceil, floor, ceil). `round_prefer_floor` is the opset-13 default but most compilers either always-floor or always-round-half-to-even. OV and ORT implement the spec literally.
- **Confidence:** high
- **Representative minimal repro:** `repros/unique_0002_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0002.py … unique_0051.py}`

---

## RC-11a · row-reduce + mul + transpose + softmax + layernorm misfold

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, torchscript, tvm, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, tensorflow, torch_compile
- **Cases (1):** unique_0004
- **Trigger pattern:** `ReduceMean(row) → Mul → Transpose → MatMul → Softmax → LayerNorm`
- **Symptom:** Four backends agree wrongly; TF+tc+ORT correct. rel_l2 = 1.0.
- **Why this is the likely root cause:** Row-reduce + transpose re-association produces different broadcast semantics.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0004_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0004.py … unique_0004.py}`

---

## RC-11b · Reciprocal(zero-input) + Conv+BN fuse corner

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, torchscript, tvm
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, tensorflow, torch_compile, xla
- **Cases (1):** unique_0007
- **Trigger pattern:** `Reciprocal → Mul → Conv → BN → Relu → Add (with a zero in denominator branch)`
- **Symptom:** 3 backends agree on a wrong non-inf value; reference produces inf/spec-correct.
- **Why this is the likely root cause:** Division-by-zero handling for Reciprocal + fused Conv/BN — these three backends replace inf with 0 post-fuse.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0007_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0007.py … unique_0007.py}`

---

## RC-11c · TF + Inductor Resize(linear, half_pixel) 5→4 shape

- **Category:** Compiler/backend bug
- **Affected backend(s):** tensorflow, torch_compile
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, tvm, openvino, xla
- **Cases (1):** unique_0014
- **Trigger pattern:** `Conv → Add → Relu → Resize(mode=linear, coord=half_pixel, H/W 5→4) → Conv`
- **Symptom:** TF + Inductor apply the same half-pixel formula disagreeing with the spec.
- **Why this is the likely root cause:** Half-pixel linear resize at non-integer ratios 5/4 exercises a different coefficient table in TF and Inductor.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0014_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0014.py … unique_0014.py}`

---

## RC-11d · branch-min-then-max + LayerNorm residual

- **Category:** Compiler/backend bug
- **Affected backend(s):** torchscript, tvm, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, tensorflow, torch_compile, openvino
- **Cases (1):** unique_0035
- **Trigger pattern:** `Max(x, c1) → Min(·, c2) → LayerNorm → Residual Add → Relu`
- **Symptom:** 3 backends diverge; rel_l2 = 1.0.
- **Why this is the likely root cause:** Saturation branch handled differently when c1/c2 overlap; cluster share TVM-style clamp lowering.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0035_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0035.py … unique_0035.py}`

---

## RC-11e · foldable add-zero / mul-one + triple MatMul constant folding

- **Category:** Compiler/backend bug
- **Affected backend(s):** tensorflow, torch_compile, torchscript
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, tvm, openvino, xla
- **Cases (1):** unique_0050
- **Trigger pattern:** `MatMul → MatMul → MatMul → Add(0) → Mul(1) → Where(const)`
- **Symptom:** 3 PyTorch/TF-side backends fold 'Add 0' and 'Mul 1' eagerly and lose the Where-const branch; rel_l2 = 1.0.
- **Why this is the likely root cause:** Trivial-identity folding removes a node whose side-effect (shape/mask) is needed downstream.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0050_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0050.py … unique_0050.py}`

---

## RC-11f · residual-add-relu + 4D matmul attention; TorchScript only correct side

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, tensorflow, torch_compile, tvm, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript
- **Cases (1):** unique_0055
- **Trigger pattern:** `Add(residual) → Relu → MatMul(4D batch) → Softmax → MatMul → Conv`
- **Symptom:** 5 backends diverge; only TS + ORT + eager correct. rel_l2 = 0.055.
- **Why this is the likely root cause:** 4D batched-matmul reshape policy differs; TS preserves onnx2torch's shape rule.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0055_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0055.py … unique_0055.py}`

---

## RC-11g · spatial-attention CBAM reduce-max ambiguity

- **Category:** Compiler/backend bug
- **Affected backend(s):** tensorflow, torchscript, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torch_compile, tvm, openvino
- **Cases (1):** unique_0065
- **Trigger pattern:** `ReduceMean → ReduceMax → Concat → Conv → Sigmoid → Mul (CBAM pattern)`
- **Symptom:** 3 backends cluster on wrong answer; rel_l2 = 1.0.
- **Why this is the likely root cause:** ReduceMax axis handling on the channel-attention branch differs.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0065_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0065.py … unique_0065.py}`

---

## RC-11h · expand+add+mul with LayerNorm axis=1 fold

- **Category:** Compiler/backend bug
- **Affected backend(s):** torch_compile, tvm
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, tensorflow, openvino, xla
- **Cases (1):** unique_0073
- **Trigger pattern:** `Expand → Add → Mul → LayerNorm(axis=1) → Conv → BN`
- **Symptom:** Inductor + TVM agree wrongly; rel_l2 = 0.05.
- **Why this is the likely root cause:** LayerNorm with axis=1 is non-canonical; both compilers attempt the same axis-rewrite and miss one dim.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0073_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0073.py … unique_0073.py}`

---

## RC-11i · expm1(bounded) + attention + resize layout

- **Category:** Compiler/backend bug
- **Affected backend(s):** torchscript, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torch_compile, tvm, openvino, tensorflow
- **Cases (1):** unique_0076
- **Trigger pattern:** `Exp → Sub(1) (manual expm1) → MatMul(4D) → Resize → Conv`
- **Symptom:** TS + XLA cluster on wrong rel_l2 = 1.0.
- **Why this is the likely root cause:** Manual expm1 pattern `Exp(x)-1` is folded to a native `expm1` with different precision rules in TS + XLA.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0076_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0076.py … unique_0076.py}`

---

## RC-11j · reciprocal-neg-zero with fp16 cast chain

- **Category:** Compiler/backend bug
- **Affected backend(s):** openvino, xla
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torchscript, torch_compile, tvm, tensorflow
- **Cases (1):** unique_0107
- **Trigger pattern:** `Relu → Mul → Reciprocal → Cast → Slice → Sub → Concat → Div → Add (with -0.0 input)`
- **Symptom:** OV + XLA give a different sign for Reciprocal(-0.0). rel_l2 = 0.09.
- **Why this is the likely root cause:** OV + XLA treat Reciprocal(-0.0) = +inf; ONNX spec and other backends return -inf.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0107_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0107.py … unique_0107.py}`

---

## RC-11k · TorchScript-only ReduceL2 + manual LayerNorm

- **Category:** Compiler/backend bug
- **Affected backend(s):** torchscript
- **Reference backend(s) (agree with pytorch_eager):** pytorch_eager, onnxruntime, torch_compile, tvm, openvino, tensorflow, xla
- **Cases (1):** unique_0108
- **Trigger pattern:** `ReduceL2 → Abs → Mul → Sqrt → Sub → Softmax (manual LayerNorm) → MatMul`
- **Symptom:** Only TorchScript diverges; rel_l2 = 1.0.
- **Why this is the likely root cause:** TorchScript's ReduceL2 path uses a different `sqrt(eps+sum_sq)` prolog than eager.
- **Confidence:** low
- **Representative minimal repro:** `repros/unique_0108_min.py`
- **Full-fidelity repros (auto-generated, per-case):** `campaign_v10_results/bugs_unique/{unique_0108.py … unique_0108.py}`

---

