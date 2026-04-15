#!/usr/bin/env python3
"""
Replay all campaign_v8_results/*_report.json through the patched oracle
scoring math and report how many would STILL be flagged as bugs.

Compares against TRIAGE.md verdict (0 real bugs, 41 false positives, 4
oracle-side crashes) to confirm the patches eliminate the FP class.

Usage:
  python tools/replay_v8_with_patched_oracle.py
"""
from __future__ import annotations
import json
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from trion.oracle.oracle import (
    _PAIRWISE_ATOL,
    _NEAR_ZERO_REF,
    _FRONTEND_KEYWORDS,
    _classify_crash,
)

CAMPAIGN_DIR = Path("campaign_v8_results")
DELTA = 0.01  # same --tolerance the v8 campaign used
BUG_THRESHOLD = 0.05  # same --bug-threshold


def patched_pairwise(
    outputs: Dict[str, np.ndarray],
    delta: float,
) -> Tuple[float, Dict[Tuple[str, str], float], Tuple[str, str] | None]:
    """Re-implement DiscrepancyOracle._oracle1_pairwise on recorded outputs."""
    names = list(outputs.keys())
    max_div = 0.0
    worst = None
    pairs: Dict[Tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            yi, yj = outputs[ni], outputs[nj]
            if yi.shape != yj.shape:
                div = 1.0
            else:
                abs_max = float(np.max(np.abs(yi - yj))) if yi.size else 0.0
                if abs_max <= _PAIRWISE_ATOL:
                    div = 0.0
                else:
                    ni_norm = float(np.linalg.norm(yi))
                    nj_norm = float(np.linalg.norm(yj))
                    max_norm = max(ni_norm, nj_norm)
                    if max_norm < _NEAR_ZERO_REF:
                        div = float(min(1.0, abs_max / (10.0 * _PAIRWISE_ATOL)))
                    else:
                        diff = float(np.linalg.norm(yi - yj))
                        raw = (diff / max_norm) / delta
                        div = float(min(1.0, raw)) if np.isfinite(raw) else 0.0
            pairs[(ni, nj)] = div
            if div > max_div:
                max_div = div
                worst = (ni, nj)
    return max_div, pairs, worst


def patched_suspect(
    outputs: Dict[str, np.ndarray],
    eager: np.ndarray | None,
    delta: float,
) -> Tuple[str | None, float]:
    """Re-implement _attribute_blame_via_eager on recorded outputs.
    Gates suspect naming on max_distance >= delta."""
    if eager is None:
        return None, 0.0
    ref = eager
    ref_norm = float(np.linalg.norm(ref))
    distances: Dict[str, float] = {}
    for name, arr in outputs.items():
        if arr.shape != ref.shape:
            distances[name] = float("inf")
            continue
        abs_max = float(np.max(np.abs(arr - ref))) if arr.size else 0.0
        if abs_max <= _PAIRWISE_ATOL:
            distances[name] = 0.0
            continue
        arr_norm = float(np.linalg.norm(arr))
        max_norm = max(ref_norm, arr_norm)
        if max_norm < _NEAR_ZERO_REF:
            distances[name] = min(1.0, abs_max / (10.0 * _PAIRWISE_ATOL))
        else:
            distances[name] = float(np.linalg.norm(arr - ref) / max_norm)
    if not distances:
        return None, 0.0
    suspect = max(distances, key=distances.get)
    dist = distances[suspect]
    if dist >= delta:
        return suspect, dist
    return None, dist  # inter-pair noise, no blame


def recover_outputs(d: dict) -> Dict[str, np.ndarray]:
    """Reconstruct per-target numpy arrays from the recorded report."""
    target_outputs = d.get("target_outputs") or {}
    out: Dict[str, np.ndarray] = {}
    for name, flat in target_outputs.items():
        if not flat:
            continue
        out[name] = np.asarray(flat, dtype=np.float64)
    return out


def recover_eager(d: dict, sample_shape) -> np.ndarray | None:
    flat = d.get("eager_output")
    if not flat:
        return None
    arr = np.asarray(flat, dtype=np.float64)
    if sample_shape is not None and arr.size == np.prod(sample_shape):
        arr = arr.reshape(sample_shape)
    return arr


def main() -> int:
    reports = sorted(glob(str(CAMPAIGN_DIR / "bug_*_report.json")))
    if not reports:
        print(f"No reports in {CAMPAIGN_DIR}", file=sys.stderr)
        return 2

    stats = {
        "total": 0,
        "still_bug": 0,
        "dropped_by_atol": 0,
        "dropped_by_suspect_gate": 0,
        "dropped_by_both": 0,
        "nonfinite_reclassified": 0,
        "tflite_reclassified": 0,
        "still_crash_backend": 0,
    }
    surviving: List[Tuple[int, float, str | None, tuple | None]] = []

    for path in reports:
        d = json.load(open(path))
        stats["total"] += 1
        model_id = d.get("model_id")

        # Reclassify recorded crashes with the patched keyword list.
        crash_reclassified_frontend = False
        for label, err in (d.get("errors") or {}).items():
            if _classify_crash(str(err)) == "frontend":
                crash_reclassified_frontend = True
                stats["tflite_reclassified"] += 1

        # "Non-finite output" crashes are now non-crashes (reassigned to a
        # per-sample nonfinite_targets entry).
        nan_crashes = [
            lbl for lbl in (d.get("crashes") or [])
            if lbl.endswith("+nan")
        ]
        if nan_crashes:
            stats["nonfinite_reclassified"] += len(nan_crashes)

        # Re-score using patched pairwise logic on the stored per-target
        # outputs.  Skip targets whose recorded output is non-finite
        # (the patched oracle now excludes them from the sample).
        target_outputs = recover_outputs(d)
        # Flatten arrays carry no shape info; pairwise only needs agreement
        # on size, which the original run already enforced.  Convert to
        # matching 1-D arrays for the atol comparison.
        target_outputs = {
            n: a for n, a in target_outputs.items()
            if np.all(np.isfinite(a))
        }

        new_score, new_pairs, new_worst = patched_pairwise(
            target_outputs, DELTA
        )
        eager = recover_eager(d, None)
        new_suspect, new_dist = patched_suspect(target_outputs, eager, DELTA)

        old_score = d.get("total_score", 0.0)
        old_suspect = d.get("suspect_backend")

        dropped_atol = old_score > BUG_THRESHOLD and new_score <= BUG_THRESHOLD
        dropped_gate = (
            old_suspect is not None
            and new_suspect is None
            and new_score > BUG_THRESHOLD
        )

        if dropped_atol and dropped_gate:
            stats["dropped_by_both"] += 1
        elif dropped_atol:
            stats["dropped_by_atol"] += 1
        elif dropped_gate:
            stats["dropped_by_suspect_gate"] += 1

        if new_score > BUG_THRESHOLD and new_suspect is not None:
            stats["still_bug"] += 1
            surviving.append((model_id, new_score, new_suspect, new_worst))

        # Remaining crash channel: a real backend crash that doesn't match
        # the new frontend keywords and isn't a NaN output.
        real_crashes = [
            lbl for lbl, info in (d.get("crash_info") or {}).items()
            if info == "backend"
            and not lbl.endswith("+nan")
            and _classify_crash(str((d.get("errors") or {}).get(lbl, ""))) == "backend"
        ]
        if real_crashes:
            stats["still_crash_backend"] += 1

    print("=" * 70)
    print("Replay of campaign_v8_results/ through patched oracle")
    print("=" * 70)
    print(f"Reports processed           : {stats['total']}")
    print(f"Still flagged as bug (new)  : {stats['still_bug']}")
    print(f"Dropped by abs-tol gating   : {stats['dropped_by_atol']}")
    print(f"Dropped by suspect-δ gate   : {stats['dropped_by_suspect_gate']}")
    print(f"Dropped by both             : {stats['dropped_by_both']}")
    print(f"NaN+crash reclassified      : {stats['nonfinite_reclassified']}")
    print(f"TFLite shape reclassified   : {stats['tflite_reclassified']}")
    print(f"Still real backend crashes  : {stats['still_crash_backend']}")
    print("-" * 70)
    if surviving:
        print("Surviving suspect reports:")
        for mid, score, sus, worst in surviving:
            print(f"  bug {mid:04d}  score={score:.3f}  "
                  f"suspect={sus}  worst_pair={worst}")
    else:
        print("No reports survive as suspect-compiler bugs.")
    return 0 if stats["still_bug"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
