#!/usr/bin/env python3
"""Re-validate bug reports using the live oracle, 3 runs for flakiness filter.

A bug is considered GENUINE if, across N repeated oracle runs on the same
model + same input, the same backend consistently shows divergence above
tolerance AND the cause is not expected quantization loss.

Usage:
    python tools/revalidate_with_oracle.py --campaign full_campaign \
        --runs 3 --workers 4 --out full_campaign/bugs_genuine
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnx

# Ensure project import path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# tflite DEFAULT optimization = post-training quantization. Divergence there
# vs pytorch_eager is expected precision loss, not a compiler bug.
QUANTIZING_BACKENDS = {"tflite"}


def _load_model_and_inputs(campaign: Path, bug_id: str):
    onnx_path = campaign / f"{bug_id}.onnx"
    report_path = campaign / f"{bug_id}_report.json"
    model = onnx.load(str(onnx_path))
    report = json.loads(report_path.read_text())
    inputs: dict[str, np.ndarray] = {}
    for name, flat in (report.get("bug_inputs") or {}).items():
        arr = np.asarray(flat, dtype=np.float32)
        # Reshape to model input shape
        for gi in model.graph.input:
            if gi.name == name:
                shape = [d.dim_value or 1 for d in gi.type.tensor_type.shape.dim]
                arr = arr.reshape(shape)
                break
        inputs[name] = arr
    return model, inputs, report


def revalidate_one(args: tuple) -> dict:
    campaign_str, bug_id, runs = args
    campaign = Path(campaign_str)
    # Import lazily in worker
    from trion.config import TrionConfig
    from trion.oracle.oracle import DiscrepancyOracle

    model, inputs, orig_report = _load_model_and_inputs(campaign, bug_id)
    cfg = TrionConfig()
    oracle = DiscrepancyOracle(cfg)

    per_run: list[dict] = []
    for _ in range(runs):
        rep = oracle.score(model, inputs, model_id=0)
        per_run.append({
            "s_diff": dict(rep.s_diff),
            "delta_opt": dict(rep.delta_opt),
            "total": rep.total_score,
        })

    # A backend is flagged if EVERY run shows s_diff or delta_opt > threshold.
    # Threshold: 0.5 means divergence is at least half the clipped range.
    threshold = 0.5
    consistent_failing: list[str] = []
    backends = set()
    for r in per_run:
        backends |= set(r["s_diff"].keys()) | set(r["delta_opt"].keys())
    for b in backends:
        always_bad = all(
            max(r["s_diff"].get(b, 0.0), r["delta_opt"].get(b, 0.0)) > threshold
            for r in per_run
        )
        if always_bad:
            consistent_failing.append(b)

    # Classify: genuine compiler bug = consistent failing backend that is not
    # purely a quantizing backend (tflite).
    non_quant = [b for b in consistent_failing if b not in QUANTIZING_BACKENDS]
    status = (
        "genuine" if non_quant
        else "quantization_only" if consistent_failing
        else "flaky_or_false_positive"
    )

    return {
        "bug_id": bug_id,
        "status": status,
        "consistent_failing": consistent_failing,
        "genuine_backends": non_quant,
        "original_s_diff": orig_report.get("s_diff", {}),
        "original_delta_opt": orig_report.get("delta_opt", {}),
        "runs": per_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="full_campaign")
    ap.add_argument("--out", default="full_campaign/bugs_genuine")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="Validate only first N bugs (0 = all)")
    args = ap.parse_args()

    campaign = Path(args.campaign)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bug_ids = sorted(
        p.stem.replace("_report", "")
        for p in campaign.glob("bug_*_report.json")
    )
    if args.limit:
        bug_ids = bug_ids[: args.limit]
    print(f"[info] {len(bug_ids)} bugs to revalidate, {args.runs} runs each")

    results: list[dict] = []
    t0 = time.time()
    tasks = [(str(campaign), b, args.runs) for b in bug_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(revalidate_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"bug_id": "?", "status": "error", "error": str(exc)[:300]})
            if i % 10 == 0 or i == len(futs):
                elapsed = time.time() - t0
                n_genuine = sum(1 for r in results if r.get("status") == "genuine")
                n_quant = sum(1 for r in results if r.get("status") == "quantization_only")
                n_flaky = sum(
                    1 for r in results if r.get("status") == "flaky_or_false_positive"
                )
                print(
                    f"[revalidate] {i}/{len(futs)} "
                    f"genuine={n_genuine} quant={n_quant} flaky={n_flaky} "
                    f"elapsed={elapsed:.0f}s"
                )

    # Write manifest
    from collections import Counter
    status_counts = Counter(r.get("status", "error") for r in results)
    by_backend: Counter = Counter()
    for r in results:
        for b in r.get("genuine_backends") or []:
            by_backend[b] += 1
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": str(campaign),
        "runs_per_bug": args.runs,
        "total_bugs": len(bug_ids),
        "status_counts": dict(status_counts),
        "genuine_by_backend": dict(by_backend),
        "results": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] manifest: {out}/manifest.json")
    print(f"[done] status: {dict(status_counts)}")
    print(f"[done] genuine by backend: {dict(by_backend)}")


if __name__ == "__main__":
    main()
