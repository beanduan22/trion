#!/usr/bin/env python3
"""
Smoke test for the four new pattern modules.

For every pattern in the 4 new modules:
  1. Try a set of synthetic ``StructuralContext``s (4-D fp32, 3-D fp32, 2-D fp32).
  2. Call ``is_compatible`` on each; skip patterns that reject every seed.
  3. For the first compatible seed, call ``instantiate`` and build a minimal
     ONNX model.
  4. Run ``onnx.checker.check_model``.
  5. Execute with ``onnxruntime`` on a random fp32 input matching the input
     context shape.

Reports per-module counts for:
    patterns_total / is_compatible_hit / instantiate_ok / onnx_check_ok / ort_run_ok

Exits 0 iff no pattern raises an unhandled exception during any of the above
stages AND the ORT run produces a finite-or-NaN array of the expected dtype.
Exits 1 otherwise.
"""
from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from trion.generation.context import StructuralContext
from trion.patterns.base import OTP, PatternInstance
from trion.patterns.cast_precision_patterns import ALL_CAST_PRECISION_PATTERNS
from trion.patterns.edge_input_patterns import ALL_EDGE_INPUT_PATTERNS
from trion.patterns.integer_arithmetic_patterns import (
    ALL_INTEGER_ARITHMETIC_PATTERNS,
)
from trion.patterns.resize_sweep_patterns import ALL_RESIZE_SWEEP_PATTERNS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke_new_patterns")


_SEED_CONTEXTS: List[Tuple[str, StructuralContext]] = [
    ("4D_NCHW", StructuralContext(4, [1, 8, 16, 16], "float32", "NCHW")),
    ("3D_NLC", StructuralContext(3, [1, 16, 64], "float32", "NLC")),
    ("2D_NC", StructuralContext(2, [1, 64], "float32", "NC")),
]


@dataclass
class ModuleStat:
    name: str
    patterns_total: int
    is_compatible_hit: int = 0
    instantiate_ok: int = 0
    onnx_check_ok: int = 0
    ort_run_ok: int = 0
    failures: List[Tuple[str, str, str]] = field(default_factory=list)


def _build_model(instance: PatternInstance, ctx: StructuralContext) -> onnx.ModelProto:
    """Wrap a PatternInstance into a single-input, single-output ONNX model."""
    input_info = helper.make_tensor_value_info(
        instance.input_name, TensorProto.FLOAT, ctx.shape
    )
    # Leave output shape unspecified — validator only checks the graph is
    # structurally consistent; ORT infers shapes at run time.
    out_ctx = instance.output_context
    output_info = helper.make_tensor_value_info(
        instance.output_name, TensorProto.FLOAT, out_ctx.shape
    )
    graph = helper.make_graph(
        instance.nodes,
        instance.pattern_name,
        [input_info],
        [output_info],
        instance.initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19)],
        ir_version=9,
    )
    return model


def _try_pattern(pattern: OTP, stat: ModuleStat) -> None:
    compatible_ctx: Optional[StructuralContext] = None
    compatible_name: str = ""
    for seed_name, seed_ctx in _SEED_CONTEXTS:
        try:
            if pattern.is_compatible(seed_ctx):
                compatible_ctx = seed_ctx
                compatible_name = seed_name
                break
        except Exception as exc:  # noqa: BLE001
            stat.failures.append((pattern.name, "is_compatible", repr(exc)))
            return
    if compatible_ctx is None:
        # Not compatible with any seed — not an error.
        return
    stat.is_compatible_hit += 1

    rng = np.random.default_rng(0xC0DE ^ hash(pattern.name) & 0xFFFF)
    try:
        instance = pattern.instantiate(
            "X", compatible_ctx, rng, node_id=1
        )
    except Exception as exc:  # noqa: BLE001
        stat.failures.append(
            (pattern.name, "instantiate", f"{exc}\n{traceback.format_exc()}")
        )
        return
    if instance is None:
        stat.failures.append(
            (pattern.name, "instantiate", "returned None despite is_compatible")
        )
        return
    stat.instantiate_ok += 1

    try:
        model = _build_model(instance, compatible_ctx)
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        stat.failures.append(
            (pattern.name, "onnx.check", f"{exc}")
        )
        return
    stat.onnx_check_ok += 1

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess = ort.InferenceSession(
            model.SerializeToString(), sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        # Sample a benign fp32 input.
        run_rng = np.random.default_rng(0xABCDEF ^ hash(pattern.name) & 0xFFFF)
        x = run_rng.standard_normal(compatible_ctx.shape).astype(np.float32)
        out = sess.run(None, {instance.input_name: x})[0]
        if not isinstance(out, np.ndarray) or out.dtype.kind == "O":
            raise RuntimeError(f"ORT produced unexpected dtype {out.dtype}")
    except Exception as exc:  # noqa: BLE001
        stat.failures.append(
            (pattern.name, "ort.run", f"{exc}")
        )
        return
    stat.ort_run_ok += 1


def main() -> int:
    modules: List[Tuple[str, List[OTP]]] = [
        ("resize_sweep_patterns", ALL_RESIZE_SWEEP_PATTERNS),
        ("integer_arithmetic_patterns", ALL_INTEGER_ARITHMETIC_PATTERNS),
        ("edge_input_patterns", ALL_EDGE_INPUT_PATTERNS),
        ("cast_precision_patterns", ALL_CAST_PRECISION_PATTERNS),
    ]
    all_stats: List[ModuleStat] = []
    any_fatal = False
    for mod_name, patterns in modules:
        stat = ModuleStat(name=mod_name, patterns_total=len(patterns))
        for pat in patterns:
            _try_pattern(pat, stat)
        all_stats.append(stat)
        if stat.failures:
            any_fatal = True

    print("=" * 74)
    print(f"{'module':<36}{'total':>6}{'comp':>6}{'inst':>6}{'chk':>6}{'run':>6}")
    print("-" * 74)
    for s in all_stats:
        print(f"{s.name:<36}"
              f"{s.patterns_total:>6}"
              f"{s.is_compatible_hit:>6}"
              f"{s.instantiate_ok:>6}"
              f"{s.onnx_check_ok:>6}"
              f"{s.ort_run_ok:>6}")
    print("=" * 74)

    if any_fatal:
        print("\nFAILURES:")
        for s in all_stats:
            for (name, stage, detail) in s.failures:
                print(f"  [{s.name}] {name}  stage={stage}")
                print(f"    {detail.splitlines()[0] if detail else ''}")
        return 1
    print("All new patterns passed is_compatible → instantiate → check → ORT run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
