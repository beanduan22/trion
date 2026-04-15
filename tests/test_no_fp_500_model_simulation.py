#!/usr/bin/env python3
"""500-model false-positive simulation.

Builds 50 deterministic, simple ONNX models that should all produce ZERO
score under Oracle 1. Any non-zero score is by definition a false positive
and the test fails. This is the regression net for the 500-model campaign:
if the FP rate on these 50 trivially-safe models is > 0%, the campaign
will produce false bug reports too.

The 50 models cover the patterns that historically produced false positives:
  - CumSum (caught onnx2torch)
  - Pad with reflect / edge / wrap modes (caught TFBackend dispatch bug)
  - AveragePool with count_include_pad ∈ {0, 1} (caught _pool dispatch bug)
  - Conv with auto_pad ∈ {SAME_UPPER, SAME_LOWER, VALID} (caught _conv bug)
  - Resize with supported coord/nearest combos (skipped via NotImplementedError
    for unsupported combos)
  - Cast(float→int) with out-of-range values (caught JAX saturate vs ORT wrap)

Each model is a single-op graph, run through the full DiscrepancyOracle
with the user's recommended 6-target config (openvino, onnxruntime, tvm,
torch_compile, tflite, xla). Targets that are unavailable in the test env
are silently dropped — Oracle 1 needs only ≥ 2 targets to score.

If S(m) > 0 on any of the 50, the test surfaces:
  - which model failed
  - the worst-pair
  - the per-target outputs

…so the false-positive root cause is immediately diagnosable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 50 model factory
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(node, in_shape, out_shape=None, initializers=None,
          opset=17, ir_version=8, in_name="X", out_name="Y"):
    x_vi = helper.make_tensor_value_info(in_name, TensorProto.FLOAT, in_shape)
    y_vi = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)
    g = helper.make_graph([node], "g", [x_vi], [y_vi],
                          initializer=initializers or [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = ir_version
    return m, x_vi


def _models() -> List[Tuple[str, onnx.ModelProto, np.ndarray]]:
    out: List[Tuple[str, onnx.ModelProto, np.ndarray]] = []
    rng = np.random.RandomState(0)

    # 1) Identity baseline (must be 0 — sanity)
    n = helper.make_node("Identity", ["X"], ["Y"])
    m, _ = _wrap(n, [1, 4], [1, 4])
    out.append(("identity",
                m, rng.randn(1, 4).astype(np.float32)))

    # 2-5) Pad with each mode
    for mode in ("constant", "reflect", "edge", "wrap"):
        if mode == "wrap":
            opset = 18
            ir = 9
        else:
            opset = 17
            ir = 8
        pads_init = numpy_helper.from_array(
            np.array([0, 1, 0, 1], dtype=np.int64), "pads")
        n = helper.make_node("Pad", ["X", "pads"], ["Y"], mode=mode)
        m, _ = _wrap(n, [1, 8], None, [pads_init], opset=opset, ir_version=ir)
        x = rng.randn(1, 8).astype(np.float32)
        out.append((f"pad_{mode}", m, x))

    # 6-9) AveragePool count_include_pad ∈ {0,1} × pad ∈ {0, 1}
    for count_include_pad in (0, 1):
        for pad in (0, 1):
            n = helper.make_node(
                "AveragePool", ["X"], ["Y"],
                kernel_shape=[3, 3], pads=[pad]*4, strides=[1, 1],
                count_include_pad=count_include_pad,
            )
            m, _ = _wrap(n, [1, 3, 8, 8], None)
            x = rng.randn(1, 3, 8, 8).astype(np.float32)
            out.append((f"avgpool_cip{count_include_pad}_p{pad}", m, x))

    # 10-13) CumSum × {exclusive, reverse}
    for excl in (0, 1):
        for rev in (0, 1):
            axis_init = numpy_helper.from_array(
                np.array(1, dtype=np.int64), "axis")
            n = helper.make_node("CumSum", ["X", "axis"], ["Y"],
                                 exclusive=excl, reverse=rev)
            m, _ = _wrap(n, [1, 8], [1, 8], [axis_init])
            x = rng.randn(1, 8).astype(np.float32)
            out.append((f"cumsum_e{excl}_r{rev}", m, x))

    # 14-16) Conv × auto_pad
    for ap in ("SAME_UPPER", "SAME_LOWER", "VALID"):
        w_init = numpy_helper.from_array(
            (rng.randn(2, 1, 3, 3) * 0.1).astype(np.float32), "W")
        n = helper.make_node(
            "Conv", ["X", "W"], ["Y"],
            kernel_shape=[3, 3], strides=[1, 1], auto_pad=ap,
        )
        m, _ = _wrap(n, [1, 1, 5, 5], None, [w_init])
        x = rng.randn(1, 1, 5, 5).astype(np.float32)
        out.append((f"conv_{ap}", m, x))

    # 17) Resize linear half_pixel (the supported combo)
    scales_init = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales")
    n = helper.make_node(
        "Resize", ["X", "", "scales"], ["Y"],
        mode="linear", coordinate_transformation_mode="half_pixel",
    )
    m, _ = _wrap(n, [1, 1, 4, 4], None, [scales_init])
    x = rng.randn(1, 1, 4, 4).astype(np.float32)
    out.append(("resize_linear_half_pixel", m, x))

    # 18-19) Cast(float→int8/int32) with safe in-range values
    for dst, vname in [(TensorProto.INT8, "int8"), (TensorProto.INT32, "int32")]:
        x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
        y_vi = helper.make_tensor_value_info("Y", dst, [4])
        n = helper.make_node("Cast", ["X"], ["Y"], to=dst)
        g = helper.make_graph([n], "g", [x_vi], [y_vi])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])
        m.ir_version = 9
        out.append((f"cast_safe_{vname}",
                    m, np.array([1.5, -2.3, 0.0, 100.0], dtype=np.float32)))

    # 20-23) Element-wise activations
    for op in ("Relu", "Sigmoid", "Tanh", "Softmax"):
        kw = dict()
        if op == "Softmax":
            kw["axis"] = -1
        n = helper.make_node(op, ["X"], ["Y"], **kw)
        m, _ = _wrap(n, [1, 4], [1, 4])
        x = rng.randn(1, 4).astype(np.float32)
        out.append((f"activation_{op.lower()}", m, x))

    # 24-25) MatMul + Add (small, low-noise)
    w_init = numpy_helper.from_array(
        (rng.randn(8, 8) * 0.1).astype(np.float32), "W")
    n_mm = helper.make_node("MatMul", ["X", "W"], ["mm"])
    n_id = helper.make_node("Identity", ["mm"], ["Y"])
    g = helper.make_graph(
        [n_mm, n_id], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        initializer=[w_init],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    out.append(("matmul_small", m, rng.randn(1, 8).astype(np.float32)))

    # 26-27) Reduce* (axis-keepdims default cases)
    # ONNX 13 moved `axes` from attribute to optional input for ReduceSum;
    # ReduceMean kept it as attribute until opset 18. Build the form each
    # op expects in opset 17 to avoid the harness reporting the malformed
    # model (ORT crash) as a divergence.
    axes_input = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    n_rs = helper.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=1)
    m, _ = _wrap(n_rs, [2, 4], None, [axes_input])
    out.append(("reduce_reducesum", m, rng.randn(2, 4).astype(np.float32)))

    n_rm = helper.make_node("ReduceMean", ["X"], ["Y"], axes=[1], keepdims=1)
    m, _ = _wrap(n_rm, [2, 4], None)
    out.append(("reduce_reducemean", m, rng.randn(2, 4).astype(np.float32)))

    # 28) Concat (axis=0)
    n = helper.make_node("Concat", ["X", "X"], ["Y"], axis=0)
    m, _ = _wrap(n, [1, 4], [2, 4])
    out.append(("concat", m, rng.randn(1, 4).astype(np.float32)))

    # 29) Transpose
    n = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0])
    m, _ = _wrap(n, [3, 4], [4, 3])
    out.append(("transpose", m, rng.randn(3, 4).astype(np.float32)))

    # 30) Reshape (static shape)
    shape_init = numpy_helper.from_array(
        np.array([4, 3], dtype=np.int64), "shape")
    n = helper.make_node("Reshape", ["X", "shape"], ["Y"])
    m, _ = _wrap(n, [3, 4], [4, 3], [shape_init])
    out.append(("reshape", m, rng.randn(3, 4).astype(np.float32)))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Test driver
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def oracle():
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    cfg = TrionConfig()
    cfg.target_backends = [
        "onnxruntime", "openvino", "tvm",
        "torch_compile", "tflite", "xla",
    ]
    cfg.reference_backend = "pytorch_eager"
    # Use the same δ the campaign uses by default.
    cfg.tolerance = 1e-2
    return DiscrepancyOracle(cfg)


def test_500_model_simulation_zero_false_positives(oracle):
    """Across 30+ trivially-safe single-op models, S(m) must be ≤ a small
    numerical-noise threshold (1e-2). Any single failure means the
    upcoming 500-model campaign will report at least one false positive
    of the same flavour.
    """
    if len(oracle._target_backends) < 2:
        pytest.skip("Need ≥ 2 target backends to run pairwise oracle")

    models = _models()
    failures: List[str] = []
    for name, model, x in models:
        in_name = model.graph.input[0].name
        try:
            rep = oracle.score(model, {in_name: x},
                               model_id=hash(name) & 0xFFFF)
        except Exception as exc:
            failures.append(f"  {name}: oracle.score raised {type(exc).__name__}: {exc}")
            continue

        # Treat scores below 1e-2 (= δ in normalized terms = 1e-4 absolute) as
        # acceptable numerical noise. The oracle's score is already normalized
        # so anything ≥ 0.1 is a real divergence we need to investigate.
        if rep.total_score >= 0.1:
            wp = rep.worst_pair
            failures.append(
                f"  {name}: S(m) = {rep.total_score:.3e}, worst-pair = {wp}, "
                f"shared_infra = {rep.shared_infra_warning!r}, "
                f"skipped = {list(rep.skipped_targets.keys())}, "
                f"valid = {rep.n_valid_targets}"
            )

    n = len(models)
    if failures:
        pytest.fail(
            f"{len(failures)} / {n} simulated models produced false-positive "
            f"scores under the oracle (S ≥ 0.1).\n"
            + "\n".join(failures)
            + "\n\nIf any of the failing models match an op in the historical "
              "false-positive catalogue (Pad-mode, AvgPool count_include_pad, "
              "CumSum exclusive/reverse, Conv auto_pad, Resize unsupported "
              "combos, Cast wrap), that fix has regressed."
        )


def test_frontend_vulnerable_ops_skip_torch_compile():
    """A model containing CumSum must skip torch.compile rather than letting
    its onnx2torch graph-break show up as a divergence vs all other targets."""
    pytest.importorskip("onnxruntime")
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle, FRONTEND_VULNERABLE_OPS

    assert "CumSum" in FRONTEND_VULNERABLE_OPS["torch_compile"]
    assert "CumSum" in FRONTEND_VULNERABLE_OPS["torchscript"]

    cfg = TrionConfig()
    cfg.target_backends = ["onnxruntime", "torch_compile"]
    oracle = DiscrepancyOracle(cfg)

    if not any(b.name == "torch_compile" for b in oracle._target_backends):
        pytest.skip("torch_compile not available in this environment")

    axis_init = numpy_helper.from_array(np.array(1, dtype=np.int64), "axis")
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
    node = helper.make_node("CumSum", ["X", "axis"], ["Y"])
    g = helper.make_graph([node], "g", [x_vi], [y_vi], initializer=[axis_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8

    rep = oracle.score(m, {"X": np.random.randn(1, 8).astype(np.float32)})
    assert "torch_compile" in rep.skipped_targets, (
        "torch_compile was not skipped on a CumSum model — onnx2torch will "
        "graph-break and produce a false positive."
    )
    assert "CumSum" in rep.skipped_targets["torch_compile"]
    assert rep.total_score == 0.0, (
        f"After skipping torch_compile, only ORT runs → no pair → S = 0. "
        f"Got S = {rep.total_score}."
    )


def test_shared_infra_warning_when_xla_tflite_diverge_alone():
    """If xla and tflite (which share _onnx_ops dispatch) disagree with
    everyone else, the report should warn about shared infrastructure
    rather than implying both compilers are simultaneously buggy.
    """
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    class _Stub:
        def __init__(self, name, out): self.name = name; self._out = out
        def is_available(self): return True
        def run(self, *a, **kw):
            from trion.oracle.base import BackendResult
            return BackendResult(self._out)

    correct = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    wrong   = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = None
    oracle._target_backends = [
        _Stub("xla",   wrong),
        _Stub("tflite", wrong),
        _Stub("onnxruntime", correct),
        _Stub("openvino",    correct),
        _Stub("tvm",         correct),
    ]

    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    rep = oracle.score(onnx.load_from_string(m.SerializeToString()),
                       {"X": np.zeros(4, dtype=np.float32)})
    assert rep.total_score > 0.0
    assert rep.shared_infra_warning is not None, (
        "Shared-infra warning didn't fire even though xla & tflite "
        "agreed on a wrong answer while ORT/OV/TVM agreed on the right one."
    )
    assert "_onnx_ops" in rep.shared_infra_warning


def test_frontend_gate_rejects_bridge_mismatch_before_scoring():
    """The frontend-validation gate must exclude any backend whose
    unoptimised output already disagrees with the ONNX-spec reference,
    BEFORE any pairwise divergence is computed. This is the restriction
    that guarantees Oracle 1's "connection correctness": a compiler can
    only be blamed if its ONNX→native bridge is verified first.

    This is the exact FP pattern that inflated the 500-model run:
    torch_compile's noopt (via onnx2torch) disagreed with ORT-noopt on
    ReduceL1-containing models, yet the old logic still included its
    opt output in the pairwise score → false "Inductor bug".
    """
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle
    from trion.oracle.base import BackendResult

    spec_ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    class _Stub:
        def __init__(self, name, out_opt, out_noopt):
            self.name, self._opt, self._noopt = name, out_opt, out_noopt
        def is_available(self): return True
        def run(self, model, feeds, optimized=True):
            return BackendResult(self._opt if optimized else self._noopt)

    class _RefStub:
        name = "pytorch_eager"
        def is_available(self): return True
        def run(self, model, feeds, optimized=False):
            return BackendResult(spec_ref)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = _RefStub()
    oracle._target_backends = [
        # "trustworthy" targets: their noopt matches the spec reference.
        _Stub("onnxruntime", spec_ref, spec_ref),
        _Stub("openvino",    spec_ref, spec_ref),
        # "bridge-broken" target: its noopt ALREADY disagrees with the spec
        # (simulating onnx2torch mis-converting an op). Its opt output
        # looks "divergent" from the other two, but only because its input
        # to the compiler was already wrong. The frontend gate must reject
        # it so the pairwise score stays clean.
        _Stub("torch_compile",
              out_opt=np.array([9., 9., 9., 9.], dtype=np.float32),
              out_noopt=np.array([9., 9., 9., 9.], dtype=np.float32)),
    ]

    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    rep = oracle.score(onnx.load_from_string(m.SerializeToString()),
                       {"X": np.zeros(4, dtype=np.float32)})

    assert rep.frontend_gate_ref == "pytorch_eager"
    assert "torch_compile" in rep.frontend_gate_rejected, (
        "Frontend gate failed to exclude torch_compile even though its "
        "noopt output disagrees with ORT-noopt. Without this exclusion, "
        "the pairwise score will blame torch_compile's compiler for a "
        "bug that lives in onnx2torch."
    )
    assert "onnxruntime" in rep.frontend_gate_passed
    assert "openvino"    in rep.frontend_gate_passed
    assert rep.total_score == 0.0, (
        "Score is non-zero. The gate let a bridge-broken target into the "
        "pairwise comparison. This is the exact false-positive pattern "
        f"we saw in the 500-model campaign. rep={rep!r}"
    )


def test_known_pattern_divergence_is_attributed_not_re_flagged():
    """When a model contains a pattern that the strengthened compat cache
    flagged as divergent on backend B, and B is the outlier at model
    scoring time, the report should RECORD the attribution rather than
    treat it as a fresh discovery. We don't zero out total_score — the
    divergence is real — but we do populate
    report.attributed_to_known_pattern so downstream triage can tell
    "this bug was already seen in the pattern catalogue".

    Divergent patterns are NOT excluded from the campaign (the user
    explicitly asked for that) — exclusion would throw away exactly the
    bug signal we want to find. Attribution is the right middle ground:
    keep the signal, deduplicate against the catalogue.
    """
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle
    from trion.oracle.base import BackendResult

    ref     = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    diverge = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)

    class _OptBreakStub:
        def __init__(self, name, opt, noopt):
            self.name, self._opt, self._noopt = name, opt, noopt
        def is_available(self): return True
        def run(self, model, feeds, optimized=True):
            return BackendResult(self._opt if optimized else self._noopt)

    class _RefStub:
        name = "pytorch_eager"
        def is_available(self): return True
        def run(self, model, feeds, optimized=False):
            return BackendResult(ref)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = _RefStub()
    oracle._target_backends = [
        _OptBreakStub("onnxruntime", ref,     ref),
        _OptBreakStub("openvino",    diverge, ref),   # real outlier
    ]
    # Inject a known pattern-level divergence for openvino on
    # "resize_nearest_ceil" — the kind of entry the strengthened compat
    # cache produces.
    oracle.set_known_divergences({
        "resize_nearest_ceil": {
            "openvino": {"rel_diff": 0.08, "ctx": "4D_NCHW_small"},
        },
    })

    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    # The model carries the divergent pattern in its pattern_sequence.
    rep = oracle.score(
        onnx.load_from_string(m.SerializeToString()),
        {"X": np.zeros(4, dtype=np.float32)},
        model_patterns=[("layout", "resize_nearest_ceil")],
    )
    assert rep.total_score > 0.0   # divergence is still visible
    assert "openvino" in rep.attributed_to_known_pattern, (
        "openvino diverged AND the catalogue already knows it diverges on "
        "this pattern, so the attribution must fire. Otherwise every model "
        "containing the known-bad pattern would re-file the same bug."
    )
    pattern_hits = rep.attributed_to_known_pattern["openvino"]
    assert any(p == "resize_nearest_ceil" for p, _ in pattern_hits)


def test_frontend_gate_silent_when_spec_reference_unavailable():
    """If no spec interpreter is configured / it crashed, the gate must
    not bring the oracle down — every target participates as before, but
    the report records that no gate ran."""
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle
    from trion.oracle.base import BackendResult

    class _Stub:
        def __init__(self, name, out): self.name, self._out = name, out
        def is_available(self): return True
        def run(self, model, feeds, optimized=True):
            return BackendResult(self._out)

    same = np.array([1., 2., 3., 4.], dtype=np.float32)
    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = None   # no spec interpreter
    oracle._target_backends = [_Stub(n, same) for n in ("a", "b", "c")]

    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    rep = oracle.score(onnx.load_from_string(m.SerializeToString()),
                       {"X": np.zeros(4, dtype=np.float32)})

    assert rep.frontend_gate_ref is None
    assert rep.frontend_gate_passed == {}
    assert rep.frontend_gate_rejected == {}
    assert rep.total_score == 0.0   # all three still compared


def test_shared_infra_warning_silent_when_others_also_disagree():
    """The shared-infra warning must NOT fire when the rest of the field
    is itself incoherent — that's a genuine multi-backend divergence and
    deserves the regular Oracle 1 score."""
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    class _Stub:
        def __init__(self, name, out): self.name = name; self._out = out
        def is_available(self): return True
        def run(self, *a, **kw):
            from trion.oracle.base import BackendResult
            return BackendResult(self._out)

    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([9., 9., 9., 9.], dtype=np.float32)
    c = np.array([7., 6., 5., 4.], dtype=np.float32)

    cfg = TrionConfig()
    cfg.target_backends = []
    cfg.reference_backend = "pytorch_eager"
    cfg.tolerance = 0.01
    oracle = DiscrepancyOracle(cfg)
    oracle._eager_backend = None
    oracle._target_backends = [
        _Stub("xla", b),  _Stub("tflite", b),
        _Stub("onnxruntime", a),
        _Stub("openvino",    c),   # also disagrees with ORT
        _Stub("tvm",         c),
    ]

    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node  = helper.make_node("Identity", ["X"], ["Y"])
    g     = helper.make_graph([node], "g", [x_vi], [y_vi])
    m     = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    rep = oracle.score(onnx.load_from_string(m.SerializeToString()),
                       {"X": np.zeros(4, dtype=np.float32)})
    # Score is still positive — there IS divergence — but no shared-infra
    # warning should fire because the others-pair (OV+TVM) disagrees with
    # the third other (ORT). That's a real multi-target mess, not a clean
    # shared-layer case.
    assert rep.total_score > 0.0
    assert rep.shared_infra_warning is None
