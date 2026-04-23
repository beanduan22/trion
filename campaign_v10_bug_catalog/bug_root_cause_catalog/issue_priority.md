# Reporting Priority

Ranked order for filing upstream. Prioritization weights:

1. **Cluster size** — larger clusters = more impact and stronger statistical signal.
2. **Nature** — `backend_bug` > `uncertain` > `spec_ambiguity` (spec issues go to `onnx/onnx`, not the backend; different reporting workflow).
3. **Confidence** — `high` reporters accept without much back-and-forth.
4. **Repro clarity** — cluster representative builds a checked ONNX model in <100 lines and runs on ORT cleanly.
5. **Single-target vs multi-target** — a clear single-backend target files faster than a "5 backends all disagree" case.

---

## P1 — file immediately (clear backend bugs, large cluster, clean repro)

| RC-ID | # Bugs | File against | Why |
|-------|--------|--------------|-----|
| **RC-01** | 41 | `openxla/xla` (primary), cc: `tensorflow/tensorflow` | 41 cases with same signature; rel_l2 up to 1.0 on a plain attention graph. XLA's HLO fusion mishandles `MatMul→Transpose→Softmax→MatMul`. Representative repro: `repros/unique_0040_min.py` (21 lines of graph, ORT-clean). |
| **RC-02** | 26 | `openxla/xla` + `pytorch/pytorch` (Inductor) | Three optimizers independently miscompile Conv+BN+Relu with a broadcast constant upstream. Two issues, one shape. Split the report: Inductor and XLA each get a tailored variant derived from `repros/unique_0064_min.py`. |
| **RC-07** | 4 | Five targets (see below) | Highest rel_l2 signal (1.0 on every case), high confidence, crisp repro (`repros/unique_0074_min.py`). Five backends independently drop the TopK(k=1) singleton axis — file cross-project: `openvinotoolkit/openvino`, `tensorflow/tensorflow`, `pytorch/pytorch` (both torch.compile and torchscript), `openxla/xla`. TVM + ORT are reference. |
| **RC-09** | 3 | `openxla/xla` | XLA-only, rel_l2 = 1.0. Clean minimal case (`repros/unique_0042_min.py`, 8 nodes). |
| **RC-04** | 5 | `pytorch/pytorch` (Inductor) | Inductor-only, reproducible in `repros/unique_0034_min.py`. |

## P2 — file after local triage (single backend, lower volume, or specific runtime)

| RC-ID | # Bugs | File against | Why wait one pass |
|-------|--------|--------------|-------------------|
| **RC-05** | 5 | `tensorflow/tensorflow` | TF-eager wrong, TF-XLA correct. Narrow a minimal case to TF eager-mode only before filing (the repro already uses the softmax-axis-0 path but worth stripping residual noise). |
| **RC-11c** | 1 | `tensorflow/tensorflow` + `pytorch/pytorch` | Half-pixel linear 5→4 formula; low volume but clear and cross-project. |
| **RC-11d** | 1 | `pytorch/pytorch` (TorchScript), `apache/tvm`, `openxla/xla` | Three-way cluster on Min/Max clamp + LN residual. |
| **RC-11f** | 1 | Five targets | OV, TF, torch.compile, TVM, XLA all wrong against TS+ORT — worth reporting but singleton. |
| **RC-11g** | 1 | `tensorflow/tensorflow`, `pytorch/pytorch` (TorchScript), `openxla/xla` | CBAM reduce-max axis, rel_l2 = 1.0. |
| **RC-11i** | 1 | `pytorch/pytorch` (TorchScript), `openxla/xla` | `Exp(x)-1` folded to `expm1`. |
| **RC-11k** | 1 | `pytorch/pytorch` (TorchScript) | Isolated ReduceL2 + manual-LN divergence. |

## P3 — needs manual confirmation before filing (spec issues or single weak signals)

| RC-ID | # Bugs | Recommended target | Why confirm first |
|-------|--------|--------------------|-------------------|
| **RC-06** | 5 | `onnx/onnx` (conformance-test request), then each backend | Every backend gives a *different* answer. Not a bug against any one backend — it's missing conformance coverage of Squeeze/Unsqueeze + bias-softmax fusion. |
| **RC-10** | 2 | `onnx/onnx` (clarify `round_prefer_floor` + add test), then TF/tc/TS/TVM/XLA | Four documented nearest-mode variants; implementations fell back to the wrong one. File a conformance-test request before filing backend issues. |
| **RC-08** | 3 | Re-run disentangled (two co-factors) | Depthwise-conv+BN folding **and** Resize rounding are both present. Split the repro into a conv-only variant and a resize-only variant before filing. |
| **RC-03** | 5 | Triage: backend maintainers | fp16 cast-roundtrip elimination may be a *valid* optimizer rewrite — not obviously a bug. Ask maintainers of OV / TF / torch.compile / XLA whether they intend to preserve the roundtrip. If yes, this becomes P1 per backend; if no, it becomes a conformance-test request. |
| **RC-11a** | 1 | — | Single case; re-confirm hypothesis before filing. |
| **RC-11b** | 1 | `onnx/onnx` | `Reciprocal(0)` + Conv+BN fuse — IEEE-754/UB corner not pinned by ONNX. Needs spec clarification, not a backend bug report. |
| **RC-11e** | 1 | `onnx/onnx` | `Add(0)`/`Mul(1)` identity-folding removes a node whose side-effect is required. Spec does not say folding is forbidden across `Where(const)`. Clarification request. |
| **RC-11h** | 1 | — | Low rel_l2 (0.053); could be fp32 round-off. Verify before filing. |
| **RC-11j** | 1 | `onnx/onnx` | `Reciprocal(-0.0)` sign — IEEE-754 corner ONNX does not require. |

---

## Summary counts

| Tier | Root-cause clusters | Total bugs in tier |
|------|----------------------|--------------------|
| P1 | 5 (RC-01, RC-02, RC-04, RC-07, RC-09) | 79 |
| P2 | 7 (RC-05, RC-11c/d/f/g/i/k) | 11 |
| P3 | 9 (RC-03, RC-06, RC-08, RC-10, RC-11a/b/e/h/j) | 20 |

**First batch to file (recommended):** RC-01 → RC-02 → RC-07 → RC-09 → RC-04. That covers 79/110 bugs (72%) and three distinct project trackers, each with a clean single-target narrative and a compact minimal repro already on disk.
