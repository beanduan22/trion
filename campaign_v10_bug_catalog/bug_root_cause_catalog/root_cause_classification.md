# Root-Cause Classification Refinement

This file refines the existing catalog **without regenerating it**. The original catalog assigned each cluster a root-cause *name*; some names (e.g. `..._ambiguity`, `..._semantics`, `..._corner`) read as if they might be either a real backend bug *or* a spec-level ambiguity. This file pins each cluster to exactly one of three buckets:

| Bucket | Meaning | Action for reporter |
|--------|---------|---------------------|
| **likely backend bug** | A small set of backends clearly disagrees with a well-defined spec/reference behaviour. File against the backend. |
| **likely specification ambiguity** | The ONNX spec admits more than one reasonable interpretation; backends implement different reasonable answers. File against `onnx/onnx` and/or ask for a conformance test. |
| **uncertain / needs manual confirmation** | Cluster size or signal is too small to commit, or the failure is plausibly a valid optimization. Needs a human-read-through before filing. |

The `nature` field has also been added to `root_cause_manifest.json` and `root_cause_summary.json` for every bug.

---

## Likely backend bugs

| RC-ID | # Bugs | Suspect backend(s) | Why this is a backend bug, not a spec issue |
|-------|--------|--------------------|---------------------------------------------|
| RC-01 | 41 | tensorflow, xla | 5 backends (ORT + TS + tc + TVM + OV) plus pytorch_eager agree within 1e-5 on a well-typed attention graph. TF+XLA share the XLA HLO backend and diverge together. That is a lowering bug, not an ambiguity. |
| RC-02 | 26 | tensorflow, torch_compile, xla | 4 spec-preserving backends agree; the 3 optimizers all miscompute Conv+BN+Relu with a broadcast constant. The ONNX Conv+BN semantics are unambiguous; this is fusion order losing fp32 precision. |
| RC-04 | 5 | torch_compile | Only Inductor diverges. |
| RC-05 | 5 | tensorflow | TF-eager wrong; TF-XLA correct. Nothing spec-level — TF runtime reordering. |
| RC-07 | 4 | openvino, tensorflow, torch_compile, torchscript, xla | TopK(k=1) shape is defined. Five backends all drop the singleton axis before LayerNorm; rel_l2 = 1.0 on every bug. TVM + ORT handle it correctly. Renamed from `..._shape_handling` → `topk_k1_singleton_axis_drop_before_layernorm` to remove ambiguity in the name. |
| RC-09 | 3 | xla | XLA-only, rel_l2 = 1.0. Not a spec issue (TF-XLA gets it right, standalone XLA does not). |
| RC-11c | 1 | tensorflow, torch_compile | Half-pixel linear coefficients differ from spec reference; OV/ORT/TS/TVM/XLA all match reference. |
| RC-11d | 1 | torchscript, tvm, xla | Clamp-then-LayerNorm-residual; 3 backends agree wrongly. |
| RC-11f | 1 | openvino, tensorflow, torch_compile, tvm, xla | 5 backends wrong against TS+ORT+eager. |
| RC-11g | 1 | tensorflow, torchscript, xla | CBAM reduce-max axis misapplied. |
| RC-11i | 1 | torchscript, xla | Manual `Exp(x)-1` folded to native `expm1` with different precision. |
| RC-11k | 1 | torchscript | TorchScript-only divergence on ReduceL2 + manual-LN. |

## Likely specification ambiguities

| RC-ID | # Bugs | Backends involved | Why this is a spec-level issue |
|-------|--------|-------------------|-------------------------------|
| RC-06 | 5 | openvino, tensorflow, torch_compile, torchscript, tvm, xla | Every compiler gives a *different* answer; only ORT + pytorch_eager (the two reference implementations) agree. The ONNX Squeeze/Unsqueeze `axes` argument moved from attribute (opset ≤ 12) to input (opset ≥ 13); when it is immediately fused with bias+softmax, reasonable implementations disagree. No single backend is "the bug." **Action:** file a conformance-test request against `onnx/onnx`. |
| RC-10 | 2 | tensorflow, torch_compile, torchscript, tvm, xla | `coordinate_transformation_mode=asymmetric` + `nearest_mode=round_prefer_floor` has four documented variants; only OV + ORT implement the literal spec. Most other backends silently fall back to `floor` or `round_half_to_even`. Documented ambiguity. |
| RC-11b | 1 | openvino, torchscript, tvm | `Reciprocal(0)` fused with Conv+BN is an IEEE-754/UB corner that ONNX does not pin down. |
| RC-11e | 1 | tensorflow, torch_compile, torchscript | `Add(0)` and `Mul(1)` identity-folding is a valid optimizer rewrite per most specs; it just happens to remove a node whose shape/mask was needed downstream. Need spec clarification on whether identity-folding may cross a `Where(const)` boundary. |
| RC-11j | 1 | openvino, xla | `Reciprocal(-0.0)` = `+inf` or `-inf`? IEEE-754 says `-inf`; ONNX does not explicitly require IEEE signed-zero semantics. |

## Uncertain / needs manual confirmation

| RC-ID | # Bugs | Why we cannot commit yet |
|-------|--------|---------------------------|
| RC-03 | 5 | `Cast(fp32→fp16) → Cast(fp16→fp32)` elimination is arguably a *valid* optimization. ONNX does not require fp16 precision loss to be preserved. Half-split between "bug" and "spec": need a maintainer of each optimizing backend to weigh in. |
| RC-08 | 3 | Two co-occurring signals (depthwise-conv+BN folding **and** nearest-asymmetric rounding). Either alone would land in a different bucket; together the cluster signature is confounded. Re-run with the two factors disentangled before filing. |
| RC-11a | 1 | Single case; hypothesis is plausible but unverified. |
| RC-11h | 1 | Single case, rel_l2 barely above tolerance (0.053). Could be fp32 round-off noise. |

---

## Renames (informative only — the RC-IDs are preserved)

These RC names in the original catalog read as ambiguity-first; the refined ones are bug-first where warranted:

| RC-ID | Original name | Refined name |
|-------|---------------|--------------|
| RC-06 | `onnx_spec_squeeze_unsqueeze_bias_softmax_fusion_ambiguity` | (unchanged — confirmed spec-ambiguity) |
| RC-07 | `topk_k1_plus_layernorm_axis_shape_handling` | `topk_k1_singleton_axis_dropped_before_layernorm` |
| RC-08 | `resize_nearest_asymmetric_or_depthwise_conv_bn_fold` | `resize_nearest_rounding_and_depthwise_bn_fold_confound` |
| RC-10 | `resize_nearest_round_prefer_floor_semantics` | `resize_nearest_round_prefer_floor_spec_ambiguity` |
| RC-11b | `reciprocal_zero_and_conv_bn_fuse_corner` | `reciprocal_of_zero_fused_with_conv_bn_uB_corner` |
| RC-11e | `foldable_add_zero_matmul_triple_constant_folding` | `identity_folding_crosses_where_const_boundary` |
| RC-11j | `reciprocal_neg_zero_with_cast_chain` | `reciprocal_of_neg_zero_sign_ieee754_corner` |

These renames are advisory; the machine-readable manifests keep the original slug for stability, but carry the `nature` field that this file defines.
