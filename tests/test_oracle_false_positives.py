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
    """Minimal backend stub used to exercise the oracle. Returns the same
    output for both optimized and noopt runs so the frontend-validation
    gate treats the stub as "bridge correct" — that lets us write tests
    that target just Oracle 1's pairwise logic without having to also
    stub out the spec reference."""

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


def _identity_model() -> onnx.ModelProto:
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    return onnx.load_from_string(m.SerializeToString())


def test_oracle1_pairwise_returns_zero_when_all_targets_agree():
    """No target disagrees → S(m) = 0. The eager helper must not fire."""
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    same = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    # Spec reference agrees with every target — gate passes, all three
    # participate, they all agree, score is 0, eager-helper never fires.
    oracle._eager_backend = _StubBackend("pytorch_eager", same)
    oracle._target_backends = [_StubBackend(n, same) for n in "abc"]

    rep = oracle.score(_identity_model(),
                       {"X": np.zeros(4, dtype=np.float32)},
                       model_id=1)
    assert rep.total_score == 0.0
    assert rep.worst_pair is None
    assert rep.suspect_backend is None
    assert rep.eager_helper_used is False   # score is 0 so no blame attribution
    assert rep.n_valid_targets == 3
    assert set(rep.frontend_gate_passed) == {"a", "b", "c"}


def test_oracle1_pairwise_fires_on_real_outlier():
    """Real outlier (one target's OPT path disagrees, but its noopt agrees
    with the spec reference — so the frontend gate passes and the
    divergence is genuinely an optimiser bug). Oracle 1 must fire and the
    eager helper must blame the outlier."""
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle
    from trion.oracle.base import BackendResult

    ref     = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    diverge = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)

    class _OptBreakStub:
        """Simulates a backend whose frontend is correct but whose optimiser
        is buggy — noopt matches spec, opt diverges. This is the exact
        scenario Oracle 1 exists to catch."""
        def __init__(self, name, opt, noopt):
            self.name, self._opt, self._noopt = name, opt, noopt
        def is_available(self): return True
        def run(self, model, feeds, optimized=True):
            return BackendResult(self._opt if optimized else self._noopt)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = _StubBackend("pytorch_eager", ref)
    oracle._target_backends = [
        _OptBreakStub("a", ref,     ref),
        _OptBreakStub("b", ref,     ref),
        # Outlier: noopt correct (passes gate), opt wrong (real optimiser bug)
        _OptBreakStub("c", diverge, ref),
    ]

    rep = oracle.score(_identity_model(),
                       {"X": np.zeros(4, dtype=np.float32)},
                       model_id=2)
    assert set(rep.frontend_gate_passed) == {"a", "b", "c"}, (
        "All three stubs' noopt matches the spec ref; all should pass the "
        f"frontend gate. Rejected: {rep.frontend_gate_rejected}"
    )
    assert rep.total_score > 0.0
    assert rep.worst_pair is not None and "c" in rep.worst_pair
    assert rep.eager_helper_used is True
    assert rep.suspect_backend == "c"


def test_oracle1_never_produces_fp_when_spec_reference_is_broken():
    """If the spec reference (ORT-noopt) is wrong for some reason, the
    frontend-validation gate will reject every target (since each target's
    noopt disagrees with the wrong spec). Oracle 1 then scores on an
    empty set of targets → S(m) = 0. No false positive is ever produced.
    This is the post-500-model-run guarantee: a wrong gate can only
    *lose* information, never fabricate a bug."""
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    same = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    wrong_spec = np.array([99., 99., 99., 99.], dtype=np.float32)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = _StubBackend("pytorch_eager", wrong_spec)
    oracle._target_backends = [_StubBackend(n, same) for n in "abcde"]

    rep = oracle.score(_identity_model(),
                       {"X": np.zeros(4, dtype=np.float32)},
                       model_id=3)
    # Every target rejected because its noopt ≠ wrong_spec.
    assert set(rep.frontend_gate_rejected) == set("abcde")
    # With nothing to compare, S(m) must be 0 — no compiler blamed.
    assert rep.total_score == 0.0
    assert rep.suspect_backend is None


def test_oracle2_crash_channel_does_not_inflate_score():
    """A crashing target must not contribute to S(m). The crash is
    recorded under report.crashes / report.crash_info instead."""
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    same = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = _StubBackend("pytorch_eager", same)
    oracle._target_backends = [
        _StubBackend("a", same),
        _StubBackend("b", same),
        _StubBackend("c", None, crashed=True),
    ]

    rep = oracle.score(_identity_model(),
                       {"X": np.zeros(4, dtype=np.float32)},
                       model_id=4)
    # Crash recorded but does not raise S(m).
    assert "c+opt" in rep.crashes
    assert rep.total_score == 0.0
    assert rep.n_valid_targets == 2


def test_oracle_xla_fallback_to_tensorflow_when_jax_missing():
    """If `xla` (JAX backend) cannot be instantiated, the oracle must
    silently substitute `tensorflow` (TF-XLA via jit_compile=True) so
    XLA coverage is preserved."""
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    cfg = TrionConfig()
    cfg.target_backends = ["xla"]
    cfg.reference_backend = "pytorch_eager"

    # Force `xla` (JAX) to look unavailable, regardless of whether JAX is
    # installed in this environment.
    from trion.oracle.xla_backend import XLABackend
    saved = XLABackend.is_available
    XLABackend.is_available = lambda self: False
    try:
        oracle = DiscrepancyOracle(cfg)
    finally:
        XLABackend.is_available = saved

    names = [b.name for b in oracle._target_backends]
    assert "tensorflow" in names, (
        f"XLA → TF-XLA fallback did not fire; targets={names}. "
        "Without the fallback, asking for `xla` on a JAX-less machine "
        "leaves the oracle with one fewer compiler than the user "
        "configured."
    )
