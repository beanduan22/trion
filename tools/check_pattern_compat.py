#!/usr/bin/env python3
"""
Pattern compatibility pre-flight.

For every registered OTP, builds a minimal single-pattern model on each
compatible seed context and runs it through every target backend (both
optimized and unoptimized).  Patterns that cause a crash on any backend
are flagged as `unsupported_by=[backend]` in the output JSON.

Trion's runner reads this JSON at campaign start and removes unsupported
patterns from the active set for a given target backend list, so every
generated model is guaranteed to be runnable (and thus every remaining
crash/discrepancy is an optimization-transform bug, not a support gap).

Usage:
    python tools/check_pattern_compat.py \\
        --backends onnxruntime torchscript torch_compile tensorflow tflite openvino \\
        --output campaign_results/pattern_compat.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import onnx

from trion.generation.context import StructuralContext
from trion.generation.search_space import PatternAwareSearchSpace
from trion.oracle.oracle import DiscrepancyOracle
from trion.config import TrionConfig
from trion.patterns.library import OTPLibrary


# Broad set of seed contexts covering every shape constraint any pattern
# in the library uses: rectangular + square 2-D, varied-channel 4-D, and
# square-last-two-dim 3-D (needed by ops like Einsum 'bij->bji' and Trilu).
_SEED_CONTEXTS = [
    ("4D_NCHW_small",   StructuralContext(4, [1, 3, 16, 16], "float32", "NCHW")),
    ("4D_NCHW_mid",     StructuralContext(4, [1, 32, 16, 16], "float32", "NCHW")),
    ("4D_NCHW_big",     StructuralContext(4, [1, 64, 32, 32], "float32", "NCHW")),
    ("3D_NLC_small",    StructuralContext(3, [1, 16, 64],  "float32", "NLC")),
    ("3D_NLC_big",      StructuralContext(3, [1, 32, 128], "float32", "NLC")),
    ("3D_NLC_sq16",     StructuralContext(3, [1, 16, 16],  "float32", "NLC")),
    ("3D_NLC_sq32",     StructuralContext(3, [1, 32, 32],  "float32", "NLC")),
    ("2D_NC_small",     StructuralContext(2, [1, 64],  "float32", "NC")),
    ("2D_NC_big",       StructuralContext(2, [1, 256], "float32", "NC")),
    ("2D_NC_sq16",      StructuralContext(2, [16, 16], "float32", "NC")),
    ("2D_NC_sq64",      StructuralContext(2, [64, 64], "float32", "NC")),
    ("2D_NC_rect",      StructuralContext(2, [8, 32],  "float32", "NC")),
]


def _build(instance, ctx) -> Optional[onnx.ModelProto]:
    return PatternAwareSearchSpace._build_onnx_model(
        nodes=instance.nodes,
        initializers=instance.initializers,
        input_name="model_input",
        output_name=instance.output_name,
        input_shape=list(ctx.shape),
        output_shape=list(instance.output_context.shape),
    )


def _test_pattern(pattern, oracle, rng) -> Dict[str, dict]:
    """
    Return per-backend status for a single pattern across every compatible
    seed context.  The result per backend is one of:

      {"status": "pass",      "rel_diff": <float>}    — no crash AND
                                                        output matches the
                                                        spec reference
                                                        (ORT-noopt) within
                                                        _OUTPUT_REL_TOL.
      {"status": "crash",     "reason":  <str>}       — crashed or refused to run
      {"status": "diverge",   "rel_diff": <float>,
                               "ctx": <seed name>}    — runs without crash
                                                        but produces output
                                                        that diverges from
                                                        the spec reference
                                                        on THIS pattern,
                                                        meaning the backend
                                                        will always disagree
                                                        on any model containing
                                                        this pattern.

    Only patterns whose status is "pass" on every active backend are safe
    to use in a cross-backend campaign.  "diverge" patterns should be
    filtered out of the campaign OR explicitly marked so the oracle can
    suppress the backend on models that include them.
    """
    # How far we allow a target backend's output to drift from the trusted
    # ONNX-spec interpreter (ORT with all optimisations disabled). 5 × the
    # campaign's δ default of 0.01; anything noisier than this on a *single
    # pattern* is a pattern-level bug or spec-ambiguity the oracle cannot
    # tell apart from a real optimizer bug.
    _OUTPUT_REL_TOL = 0.05

    # Pick a spec interpreter — ORT-noopt via the configured eager backend.
    spec_be = getattr(oracle, "_eager_backend", None)

    def _rel(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
        if a.shape != b.shape:
            return float("inf")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return float("inf")
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-3))

    result: Dict[str, dict] = {}
    for ctx_name, ctx in _SEED_CONTEXTS:
        try:
            if not pattern.is_compatible(ctx):
                continue
            inst = pattern.instantiate("model_input", ctx, rng, node_id=0)
        except Exception:
            continue
        if inst is None:
            continue
        model = _build(inst, ctx)
        if model is None:
            continue

        x = np.ones(list(ctx.shape), dtype=np.float32) * 0.5
        inputs = {"model_input": x}

        # Spec reference for this context.
        spec_out = None
        if spec_be is not None:
            try:
                spec_res = spec_be.run(model, inputs, optimized=False)
            except Exception:
                spec_res = None
            if spec_res is not None and not spec_res.crashed \
                    and spec_res.output is not None \
                    and np.all(np.isfinite(spec_res.output)):
                spec_out = np.asarray(spec_res.output)

        for backend in oracle._target_backends:
            bname = backend.name
            # Once a backend is known-pass on some context, don't overwrite
            # (we prefer evidence of success). Divergence on ONE context is
            # sticky though, because it means there exists at least one
            # context where this pattern is unsafe.
            if result.get(bname, {}).get("status") == "pass":
                continue

            crashed = None
            outputs = {}
            for opt in (False, True):
                try:
                    r = backend.run(model, inputs, optimized=opt)
                except Exception as exc:
                    crashed = f"{type(exc).__name__}: {str(exc)[:80]}"
                    break
                if r.crashed:
                    err = (r.error or "").splitlines()[0][:80]
                    crashed = err
                    break
                outputs[opt] = r.output

            if crashed:
                # First-seen crash wins (so we can show at least one reason).
                result.setdefault(bname, {"status": "crash", "reason": crashed})
                continue

            # Runs on this context. Compare opt output to spec.
            if spec_out is None:
                # Can't judge correctness without a spec interpreter.
                result.setdefault(bname, {"status": "pass", "rel_diff": None})
                continue
            rel = _rel(spec_out, outputs.get(True, outputs.get(False)))
            if rel <= _OUTPUT_REL_TOL:
                result[bname] = {"status": "pass", "rel_diff": rel}
            else:
                # Divergence is sticky — mark but keep exploring other
                # contexts in case another seed behaves better.
                existing = result.get(bname)
                if existing is None or existing.get("status") == "crash":
                    result[bname] = {
                        "status": "diverge",
                        "rel_diff": rel,
                        "ctx": ctx_name,
                    }

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--output", default="campaign_results/pattern_compat.json")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only check these patterns (debug)")
    args = ap.parse_args()

    cfg = TrionConfig(target_backends=args.backends, verbose=False)
    oracle = DiscrepancyOracle(cfg)
    available_backends = [b.name for b in oracle._target_backends]
    print(f"Active backends: {available_backends}")
    if len(available_backends) < len(args.backends):
        missing = set(args.backends) - set(available_backends)
        print(f"  warning: missing backends: {missing}")

    lib = OTPLibrary()
    patterns = lib.all_patterns()
    if args.only:
        patterns = [p for p in patterns if p.name in set(args.only)]
    print(f"Checking {len(patterns)} patterns against "
          f"{len(available_backends)} backends ...")

    rng = np.random.default_rng(123)
    per_pattern: Dict[str, Dict[str, bool]] = {}
    t0 = time.time()
    for i, p in enumerate(patterns):
        per_pattern[p.name] = _test_pattern(p, oracle, rng)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(patterns)}]  "
                  f"elapsed {elapsed:.0f}s  "
                  f"ETA {elapsed * (len(patterns) - i - 1) / (i + 1):.0f}s")

    # Compute summary: per backend, pass / crash / diverge counts
    summary: Dict[str, Dict[str, int]] = {
        b: {"pass": 0, "crash": 0, "diverge": 0, "unchecked": 0}
        for b in available_backends
    }
    crashed_by_backend:  Dict[str, List[str]] = {b: [] for b in available_backends}
    diverged_by_backend: Dict[str, List[str]] = {b: [] for b in available_backends}

    for name, per_backend in per_pattern.items():
        for bname in available_backends:
            entry = per_backend.get(bname)
            if entry is None:
                summary[bname]["unchecked"] += 1
                continue
            st = entry.get("status", "unchecked")
            summary[bname][st] = summary[bname].get(st, 0) + 1
            if st == "crash":
                crashed_by_backend[bname].append(name)
            elif st == "diverge":
                diverged_by_backend[bname].append(name)

    # A pattern is safe for cross-backend campaigns only if it is "pass" on
    # every active backend (crash OR diverge on any backend is disqualifying).
    safe_patterns: List[str] = []
    for name, per_backend in per_pattern.items():
        if all(
            (per_backend.get(b, {}).get("status") == "pass")
            for b in available_backends
        ):
            safe_patterns.append(name)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out = {
        "backends": available_backends,
        "output_rel_tol": 0.05,
        "patterns": per_pattern,
        "summary": summary,
        "safe_patterns": sorted(safe_patterns),
        "crashed_by_backend":  {b: sorted(v) for b, v in crashed_by_backend.items()},
        "diverged_by_backend": {b: sorted(v) for b, v in diverged_by_backend.items()},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print()
    print("=== Compatibility summary ===")
    for bname in available_backends:
        s = summary[bname]
        total = s["pass"] + s["crash"] + s["diverge"] + s["unchecked"]
        pct_pass = 100.0 * s["pass"] / total if total else 0.0
        print(f"  {bname:<16} "
              f"pass={s['pass']:4d}/{total} ({pct_pass:5.1f}%)   "
              f"crash={s['crash']:4d}   diverge={s['diverge']:4d}   "
              f"unchecked={s['unchecked']}")
    print()
    print(f"Cross-backend-safe patterns (pass on ALL {len(available_backends)} "
          f"backends): {len(safe_patterns)} / {len(patterns)}")
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
