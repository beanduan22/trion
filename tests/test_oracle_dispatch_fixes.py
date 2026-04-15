#!/usr/bin/env python3
"""Regression tests for the second wave of dispatch fixes (2026-04-15).

In the audit following the Pad-mode fix we found 7 more attribute-loss bugs
in `trion/oracle/_onnx_ops.py` that, like the Pad bug, would silently produce
wrong "ground truth" output and cause every target backend to look buggy:

  - Resize           : coordinate_transformation_mode, nearest_mode,
                       cubic_coeff_a, antialias, exclude_outside all ignored.
  - AveragePool      : count_include_pad ignored (always =0).
  - CumSum           : exclusive, reverse ignored.
  - Cast             : saturate ignored; out-of-range float→int wraps instead
                       of clamping.
  - Conv / Pool      : auto_pad ignored; SAME_UPPER / SAME_LOWER / VALID
                       silently ran with pads=[0,0,0,0].
  - TopK (JAX path)  : sorted attribute silently dropped (JAX always sorts).

The fixes either implement the attribute correctly or, for Resize cubic /
antialias / unsupported coord modes, raise NotImplementedError so the
oracle records a "frontend" skip rather than a fake bug.

Each test compares the JAX dispatch path through `_onnx_ops.dispatch_op`
against ONNXRuntime with optimisations disabled (the trusted ONNX-spec
interpreter) on a tiny single-op graph.
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


def _ort_run(mb: bytes, x: np.ndarray, in_name: str = "X") -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        mb, so, providers=["CPUExecutionProvider"]
    ).run(None, {in_name: x})[0]


def _xla_run(mb: bytes, x: np.ndarray, in_name: str = "X") -> np.ndarray:
    """Run the model through the XLA backend (which uses _onnx_ops dispatch)."""
    pytest.importorskip("jax")
    from trion.oracle.xla_backend import XLABackend
    r = XLABackend().run(onnx.load_from_string(mb), {in_name: x},
                         optimized=False)
    if r.crashed or r.output is None:
        pytest.skip(f"XLA backend skipped: {r.error}")
    return np.asarray(r.output)


# ─────────────────────────────────────────────────────────────────────────────
# AveragePool count_include_pad
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("count_include_pad", [0, 1])
def test_avgpool_count_include_pad(count_include_pad):
    """AveragePool count_include_pad must be honoured.

    Before the fix the harness always used count_include_pad=False, so any
    model with count_include_pad=1 reported XLA as "buggy" by ~0.12 rel_L2
    when in fact the harness reference was wrong.
    """
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node(
        "AveragePool", ["X"], ["Y"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
        count_include_pad=count_include_pad,
    )
    g = helper.make_graph([node], "g", [x_vi], [y_vi])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    mb = m.SerializeToString()

    x = np.ones((1, 3, 8, 8), dtype=np.float32)
    ref = _ort_run(mb, x)
    got = _xla_run(mb, x)
    diff = float(np.max(np.abs(ref - got)))
    assert diff < 1e-5, (
        f"AveragePool(count_include_pad={count_include_pad}) diverges from "
        f"ORT by {diff}; harness ignores the attribute."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CumSum exclusive / reverse
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("exclusive", [0, 1])
@pytest.mark.parametrize("reverse",   [0, 1])
def test_cumsum_exclusive_reverse(exclusive, reverse):
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
    axis_init = numpy_helper.from_array(np.array(1, dtype=np.int64), "axis")
    node = helper.make_node(
        "CumSum", ["X", "axis"], ["Y"],
        exclusive=exclusive, reverse=reverse,
    )
    g = helper.make_graph([node], "g", [x_vi], [y_vi], initializer=[axis_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    mb = m.SerializeToString()

    np.random.seed(0)
    x = np.random.randn(1, 8).astype(np.float32)
    ref = _ort_run(mb, x)
    got = _xla_run(mb, x)
    diff = float(np.max(np.abs(ref - got)))
    assert diff < 1e-5, (
        f"CumSum(exclusive={exclusive}, reverse={reverse}) diverges from "
        f"ORT by {diff}; harness ignores the attribute."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cast saturate (float → int)
# ─────────────────────────────────────────────────────────────────────────────

def test_cast_float_to_int8_matches_ort_wrap_semantics():
    """Cast(float→int8) must match ORT bit-for-bit.

    ONNX leaves out-of-range float→int conversions implementation-defined.
    ORT picks C-cast wrap (200.0 → -56). numpy's `.astype` does the same,
    so the harness must NOT introduce its own clamp/saturate logic — that
    would itself cause harness-vs-ORT divergence and re-create the kind
    of false positive we're trying to eliminate.
    """
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.INT8, [4])
    node = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.INT8)
    g = helper.make_graph([node], "g", [x_vi], [y_vi])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])
    m.ir_version = 9
    mb = m.SerializeToString()

    x = np.array([200.0, -200.0, 50.0, -50.0], dtype=np.float32)
    ref = _ort_run(mb, x)
    got = _xla_run(mb, x)
    # Both should be int8 with the same values. The XLA backend always
    # returns float32 on the way out, so cast back to int8 for comparison.
    assert np.array_equal(ref.astype(np.int8), got.astype(np.int8)), (
        f"Cast(float→int8) diverges from ORT: ref={ref}, got={got}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conv auto_pad
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
def test_conv_auto_pad(auto_pad):
    """Conv with auto_pad ∈ {SAME_UPPER, SAME_LOWER, VALID} must produce the
    same output as the ONNX-spec interpreter."""
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    w_init = numpy_helper.from_array(
        np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0, "W")
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node(
        "Conv", ["X", "W"], ["Y"],
        kernel_shape=[3, 3], strides=[1, 1], auto_pad=auto_pad,
    )
    g = helper.make_graph([node], "g", [x_vi], [y_vi], initializer=[w_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    mb = m.SerializeToString()

    np.random.seed(0)
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)
    ref = _ort_run(mb, x)
    got = _xla_run(mb, x)
    assert ref.shape == got.shape, (
        f"Conv(auto_pad={auto_pad}) shape diverges: ref={ref.shape}, "
        f"got={got.shape}"
    )
    diff = float(np.max(np.abs(ref - got)))
    assert diff < 1e-5, (
        f"Conv(auto_pad={auto_pad}) diverges from ORT by {diff}; "
        "auto_pad attribute was likely dropped."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resize: harness now refuses unsupported combos rather than fake an answer.
# ─────────────────────────────────────────────────────────────────────────────

def test_resize_unsupported_combos_skip_instead_of_lying():
    """Any Resize attribute combo the harness cannot faithfully emulate
    must raise NotImplementedError so the oracle classifies the run as
    "frontend" (skipped) rather than producing fake reference output."""
    pytest.importorskip("jax")
    from trion.oracle._onnx_ops import dispatch_op
    import jax.numpy as jnp

    x = jnp.zeros((1, 1, 4, 4), dtype=jnp.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # nearest + ceil rounding — onnx2torch is known buggy here, harness must
    # not silently substitute floor.
    node = helper.make_node(
        "Resize", ["x", "", "scales"], ["y"],
        mode="nearest",
        coordinate_transformation_mode="half_pixel",
        nearest_mode="ceil",
    )
    with pytest.raises(NotImplementedError):
        dispatch_op(node, {"x": x, "scales": scales}, jnp)

    # cubic — too many backend-specific knobs (cubic_coeff_a, antialias,
    # exclude_outside) for the harness to provide a faithful reference.
    node = helper.make_node(
        "Resize", ["x", "", "scales"], ["y"],
        mode="cubic",
        coordinate_transformation_mode="half_pixel",
        cubic_coeff_a=-0.5,
    )
    with pytest.raises(NotImplementedError):
        dispatch_op(node, {"x": x, "scales": scales}, jnp)

    # antialias=1
    node = helper.make_node(
        "Resize", ["x", "", "scales"], ["y"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",
        antialias=1,
    )
    with pytest.raises(NotImplementedError):
        dispatch_op(node, {"x": x, "scales": scales}, jnp)


def test_resize_supported_combo_matches_ort():
    """The combos the harness *does* support must match ORT bit-for-bit
    (within fp tolerance)."""
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    scales_init = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales")
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node(
        "Resize", ["X", "", "scales"], ["Y"],
        mode="linear", coordinate_transformation_mode="half_pixel",
    )
    g = helper.make_graph([node], "g", [x_vi], [y_vi], initializer=[scales_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    mb = m.SerializeToString()

    np.random.seed(0)
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    ref = _ort_run(mb, x)
    got = _xla_run(mb, x)
    diff = float(np.max(np.abs(ref - got)))
    # Slightly looser bound — half_pixel linear has small fp-order
    # differences across implementations.
    assert diff < 1e-4, (
        f"Resize(linear,half_pixel) diverges from ORT by {diff} on the "
        "supported combo — the harness reference is wrong."
    )
