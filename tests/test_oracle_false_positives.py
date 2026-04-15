#!/usr/bin/env python3
"""Regression tests for the false-positive sources discovered on 2026-04-15.

Three classes of false positive previously inflated the bug count:

1. TFBackend Pad-mode bug — `tf.pad(x, paddings)` was called without the
   `mode` argument, so every ONNX `Pad(mode='reflect'|'edge')` silently became
   a zero-pad. That alone produced 15 "TF-only rel_L2 ≈ 0.33" false positives
   in the campaign sweep.

2. PyTorchEagerBackend used onnx2torch as the *primary* reference. When
   onnx2torch silently mishandles an op (CumSum graph break, Resize
   nearest_ceil → floor, …) every target backend looks wrong against it.
   That produced 19 "all-backends-agree" false positives.

3. The oracle had no consensus check — a single wrong reference was always
   trusted. We now demote a reference whose answer disagrees with ≥3 mutually
   agreeing targets.

These tests are intentionally hermetic — they only require numpy + onnx +
onnxruntime + tensorflow. They do NOT require trion to be importable as an
installed package; they manipulate sys.path directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_pad_model(mode: str, shape=(1, 32), pad_each_side=2) -> bytes:
    """Single-node Pad graph along the last axis."""
    rank = len(shape)
    pads = [0] * rank * 2
    pads[rank - 1] = pad_each_side          # last-axis begin
    pads[2 * rank - 1] = pad_each_side      # last-axis end
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    pads_init = numpy_helper.from_array(np.array(pads, dtype=np.int64), "pads")
    node = helper.make_node("Pad", ["X", "pads"], ["Y"], mode=mode)
    g = helper.make_graph([node], "g", [x_vi], [y_vi], initializer=[pads_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m.SerializeToString()


def _ort_run(model_bytes: bytes, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model_bytes, so, providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})[0]


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: TFBackend honours ONNX Pad mode
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["constant", "reflect", "edge"])
def test_tf_backend_pad_mode_matches_ort(mode):
    """trion.oracle.tf_backend must pass `mode` through to tf.pad.

    Before the fix, mode='reflect' / 'edge' silently became zero-pad and
    produced rel_L2 ≈ 0.33 vs ORT — the cause of 15 historical false
    positives.
    """
    pytest.importorskip("tensorflow")
    from trion.oracle.tf_backend import TFBackend

    np.random.seed(0)
    x = np.random.randn(1, 32).astype(np.float32)
    mb = _build_pad_model(mode=mode, shape=(1, 32), pad_each_side=2)

    ref = _ort_run(mb, x).ravel()
    r = TFBackend().run(onnx.load_from_string(mb), {"X": x}, optimized=False)
    assert r.output is not None, f"TFBackend crashed on Pad(mode={mode}): {r.error}"
    diff = float(np.max(np.abs(r.output.ravel() - ref)))
    assert diff < 1e-6, (
        f"TFBackend Pad(mode={mode}) diverges from ORT by {diff}; "
        "tf_backend is dropping the `mode` argument or implementing it "
        "differently from the ONNX spec."
    )


def test_tf_backend_pad_constant_value():
    """Pad-constant must respect the `constant_value` input, not always 0."""
    pytest.importorskip("tensorflow")
    from trion.oracle.tf_backend import TFBackend

    rank = 2
    pads = [0, 1, 0, 1]
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    pads_init = numpy_helper.from_array(np.array(pads, dtype=np.int64), "pads")
    cval_init = numpy_helper.from_array(np.array(7.0, dtype=np.float32), "cval")
    node = helper.make_node("Pad", ["X", "pads", "cval"], ["Y"], mode="constant")
    g = helper.make_graph([node], "g", [x_vi], [y_vi],
                          initializer=[pads_init, cval_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    mb = m.SerializeToString()

    x = np.zeros((1, 4), dtype=np.float32)
    r = TFBackend().run(onnx.load_from_string(mb), {"X": x}, optimized=False)
    assert r.output is not None
    out = r.output.ravel()
    # First and last column must equal the constant_value (7.0).
    assert out[0] == pytest.approx(7.0)
    assert out[-1] == pytest.approx(7.0)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Reference backend prefers ORT-noopt over onnx2torch
# ─────────────────────────────────────────────────────────────────────────────

def test_reference_backend_uses_ort_first():
    """PyTorchEagerBackend must call ORT-noopt before onnx2torch.

    Before the fix, onnx2torch was tried first; its CumSum/Pad-reflect/
    Resize-nearest_ceil bugs poisoned every comparison.
    """
    pytest.importorskip("onnxruntime")
    from trion.oracle.pytorch_backend import PyTorchEagerBackend

    # Build a CumSum model — onnx2torch is known-buggy on this op.
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
    axis_init = numpy_helper.from_array(np.array(1, dtype=np.int64), "axis")
    node = helper.make_node("CumSum", ["X", "axis"], ["Y"])
    g = helper.make_graph([node], "g", [x_vi], [y_vi], initializer=[axis_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    mb = m.SerializeToString()

    np.random.seed(0)
    x = np.random.randn(1, 8).astype(np.float32)
    expected = np.cumsum(x, axis=1)

    be = PyTorchEagerBackend()
    r = be.run(onnx.load_from_string(mb), {"X": x}, optimized=False)
    assert r.output is not None and not r.crashed
    diff = float(np.max(np.abs(r.output.ravel() - expected.ravel())))
    assert diff < 1e-5, (
        f"Reference produced wrong CumSum (max_diff={diff}); the backend "
        "is still using onnx2torch as the primary reference."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Oracle consensus check demotes a wrong reference
# ─────────────────────────────────────────────────────────────────────────────

class _StubBackend:
    """Minimal backend stub used to exercise the consensus check."""

    def __init__(self, name: str, output: np.ndarray | None,
                 crashed: bool = False) -> None:
        self.name = name
        self._output = output
        self._crashed = crashed

    def is_available(self) -> bool:
        return True

    def run(self, model, inputs, optimized: bool = True):
        from trion.oracle.base import BackendResult
        if self._crashed:
            return BackendResult(None, "stub crash", crashed=True)
        return BackendResult(self._output)


def test_oracle_consensus_check_demotes_wrong_reference():
    """When ≥3 targets agree but the reference disagrees, the reference
    should be marked as the buggy party — every target's s_diff becomes 0.
    """
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    correct = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    wrong   = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)

    cfg = TrionConfig()
    cfg.target_backends = []          # we'll inject stubs by hand
    cfg.reference_backend = "pytorch_eager"
    oracle = DiscrepancyOracle(cfg)
    oracle._ref_backend = _StubBackend("pytorch_eager", wrong)
    oracle._target_backends = [
        _StubBackend("a", correct),
        _StubBackend("b", correct),
        _StubBackend("c", correct),
    ]

    # Build a trivial valid ONNX model — content is irrelevant; the stubs
    # ignore it.
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mb    = onnx.load_from_string(m.SerializeToString())

    rep = oracle.score(mb, {"X": np.zeros(4, dtype=np.float32)}, model_id=42)

    assert rep.reference_likely_wrong is True, (
        "Consensus check failed to demote a reference that disagrees with "
        "3 mutually agreeing targets."
    )
    assert rep.consensus_size == 3
    # No target should be flagged as buggy when the reference is the bug.
    assert rep.total_score == 0.0
    for name in ("a", "b", "c"):
        assert rep.s_diff[name] == 0.0, (
            f"Target {name} was scored as buggy even though the reference "
            "lost the consensus vote."
        )


def test_oracle_consensus_check_does_not_fire_when_targets_disagree():
    """When targets disagree among themselves, the reference is trusted —
    a real per-target divergence must still register as a bug.
    """
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    ref     = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    matches = ref
    diverge = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._ref_backend = _StubBackend("pytorch_eager", ref)
    oracle._target_backends = [
        _StubBackend("a", matches),
        _StubBackend("b", matches),
        _StubBackend("c", diverge),  # genuine outlier
    ]

    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mb    = onnx.load_from_string(m.SerializeToString())

    rep = oracle.score(mb, {"X": np.zeros(4, dtype=np.float32)}, model_id=7)
    assert rep.reference_likely_wrong is False
    assert rep.s_diff["a"] == 0.0
    assert rep.s_diff["b"] == 0.0
    assert rep.s_diff["c"] > 0.0, (
        "Real outlier 'c' was incorrectly suppressed; consensus check is "
        "over-eager."
    )
