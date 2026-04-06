#!/usr/bin/env python3
"""
Trion Bug Reproducer
====================
Reproduces all confirmed real compiler bugs found in the Trion fuzzing campaign.
Non-compiler crashes (onnx2torch, FakeTensor, dynamo tracing artifacts) are excluded.

Bugs are stored in bugs/ — each bug has:
  bugs/bug_NNNN.onnx          — minimal ONNX model that triggers the bug
  bugs/bug_NNNN_report.json   — compiler, patterns, error, delta scores
  bugs/index.json             — searchable index of all bugs

Usage:
    # Reproduce every bug (shows pass/fail per compiler)
    python reproduce_bugs.py

    # Reproduce bugs for one compiler class
    python reproduce_bugs.py --class onnxruntime
    python reproduce_bugs.py --class torchscript
    python reproduce_bugs.py --class torch_compile
    python reproduce_bugs.py --class xla
    python reproduce_bugs.py --class tvm
    python reproduce_bugs.py --class tflite
    python reproduce_bugs.py --class tensorflow
    python reproduce_bugs.py --class openvino

    # Reproduce a single bug by ID
    python reproduce_bugs.py --bug bug_0007

    # List all bugs without running
    python reproduce_bugs.py --list

    # List bugs for one class
    python reproduce_bugs.py --list --class xla

    # Verbose: print full error on failure
    python reproduce_bugs.py --verbose
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx

BUGS_DIR  = os.path.join(os.path.dirname(__file__), "bugs")
DISC_THRESH = 0.05   # relative-difference threshold for discrepancy bugs

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


# ── backend loader ────────────────────────────────────────────────────────────

def _load_backend(name: str):
    """Import and return the BackendBase subclass for *name*."""
    from trion.config import TrionConfig
    cfg = TrionConfig(num_models=1, target_backends=[name])
    if name == "pytorch_eager":
        from trion.oracle.pytorch_backend import PyTorchEagerBackend
        return PyTorchEagerBackend()
    if name == "onnxruntime":
        from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
        return ONNXRuntimeBackend()
    if name == "torchscript":
        from trion.oracle.torchscript_backend import TorchScriptBackend
        return TorchScriptBackend()
    if name == "torch_compile":
        from trion.oracle.torch_compile_backend import TorchCompileBackend
        return TorchCompileBackend()
    if name == "xla":
        from trion.oracle.xla_backend import XLABackend
        return XLABackend(cfg)
    if name == "tvm":
        from trion.oracle.tvm_backend import TVMBackend
        return TVMBackend(cfg)
    if name == "tensorflow":
        from trion.oracle.tf_backend import TFBackend
        return TFBackend(cfg)
    if name == "tflite":
        from trion.oracle.tflite_backend import TFLiteBackend
        return TFLiteBackend(cfg)
    if name == "openvino":
        from trion.oracle.openvino_backend import OpenVINOBackend
        return OpenVINOBackend(cfg)
    raise ValueError(f"Unknown backend: {name!r}")


# ── bug loading ───────────────────────────────────────────────────────────────

def load_bugs(
    bug_filter: Optional[str] = None,
    class_filter: Optional[str] = None,
) -> List[dict]:
    index_path = os.path.join(BUGS_DIR, "index.json")
    if not os.path.exists(index_path):
        sys.exit(f"Bug index not found: {index_path}\nRun Trion first to generate bugs/")

    with open(index_path) as fp:
        index = json.load(fp)

    results = []
    for entry in index:
        bid = entry["bug_id"]
        if bug_filter and bid != bug_filter:
            continue
        if class_filter and class_filter not in entry["compilers"]:
            continue
        report_path = os.path.join(BUGS_DIR, f"{bid}_report.json")
        onnx_path   = os.path.join(BUGS_DIR, f"{bid}.onnx")
        if not os.path.exists(report_path) or not os.path.exists(onnx_path):
            continue
        with open(report_path) as fp:
            report = json.load(fp)
        report["_onnx"] = onnx_path
        results.append(report)
    return results


# ── input generation ──────────────────────────────────────────────────────────

def make_inputs(model: onnx.ModelProto, shape: list) -> Dict[str, np.ndarray]:
    name = model.graph.input[0].name
    rng  = np.random.default_rng(42)
    return {name: rng.standard_normal(shape).astype(np.float32)}


# ── relative difference ───────────────────────────────────────────────────────

def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.linalg.norm(a.flatten() - b.flatten())
    norm = max(np.linalg.norm(a.flatten()), 1e-8)
    return float(diff / norm)


# ── reproduce one bug ─────────────────────────────────────────────────────────

def reproduce(report: dict, verbose: bool = False) -> Tuple[bool, str]:
    """
    Load the ONNX model, run it through the affected compilers, and verify
    that the crash or discrepancy is still present.

    Returns (confirmed: bool, summary: str).
    """
    bid        = report["bug_id"]
    onnx_path  = report["_onnx"]
    compilers  = report["affected_compilers"]
    ishape     = report.get("input_shape", [1, 3, 32, 32])
    patterns   = " → ".join(p[1] for p in report.get("pattern_sequence", []))
    crashes    = set(report.get("crashes", []))
    delta_opt  = report.get("delta_opt", {})
    saved_errs = report.get("errors", {})

    model  = onnx.load(onnx_path)
    inputs = make_inputs(model, ishape)

    # Reference: PyTorch eager
    try:
        ref_be  = _load_backend("pytorch_eager")
        ref_res = ref_be.run(model, inputs, optimized=False)
    except Exception as exc:
        return False, f"[SKIP] reference backend error: {exc}"

    if ref_res.crashed or ref_res.output is None:
        return False, f"[SKIP] reference (pytorch_eager) crashed: {ref_res.error[:120]}"

    ref_out = ref_res.output

    confirmed = False
    lines: List[str] = [
        f"Bug {bid}  score={report['total_score']:.4f}  class={report['compiler_class']}",
        f"  patterns : {patterns}",
        f"  input    : {ishape}",
        f"  compilers: {', '.join(compilers)}",
    ]

    for compiler in compilers:
        if compiler == "pytorch_eager":
            continue
        try:
            be = _load_backend(compiler)
        except Exception as exc:
            lines.append(f"  [{compiler}] SKIP (not installed: {exc})")
            continue

        for opt in [True, False]:
            tag = f"{compiler}+{'opt' if opt else 'noopt'}"
            crash_key_opt = f"{compiler}+opt"
            crash_key_no  = f"{compiler}-opt"

            try:
                res = be.run(model, inputs, optimized=opt)
            except Exception as exc:
                res_err = str(exc)
                lines.append(f"  [{tag}] EXCEPTION: {res_err[:120]}")
                confirmed = True
                continue

            if res.crashed:
                err_short = (res.error or "")[:120]
                lines.append(f"  [{tag}] ✗ CRASH: {err_short}")
                if verbose and res.error:
                    lines.append(f"           full error:\n{res.error}")
                confirmed = True
            elif res.output is None:
                lines.append(f"  [{tag}] ✗ NO OUTPUT (no crash, no result)")
                confirmed = True
            else:
                rd = rel_diff(ref_out, res.output)
                expected_disc = delta_opt.get(compiler, 0) > DISC_THRESH
                if rd > DISC_THRESH:
                    lines.append(f"  [{tag}] ✗ WRONG  rel_diff={rd:.4f}")
                    confirmed = True
                else:
                    mark = "✓ OK" if not expected_disc else "? DISC-GONE"
                    lines.append(f"  [{tag}] {mark}  rel_diff={rd:.4f}")

            # Only run no-opt for compilers that are expected to show opt-vs-noopt diff
            if not opt and delta_opt.get(compiler, 0) <= DISC_THRESH and compiler not in {
                c.replace("+opt","").replace("-opt","") for c in crashes
            }:
                break

    status = "CONFIRMED" if confirmed else "NOT REPRODUCED"
    lines.append(f"  → {status}")
    return confirmed, "\n".join(lines)


# ── list ──────────────────────────────────────────────────────────────────────

def list_bugs(bugs: List[dict]) -> None:
    COMPILERS = ["onnxruntime","torchscript","torch_compile","xla","tvm",
                 "tensorflow","tflite","openvino"]
    header = f"{'Bug ID':<12} {'Score':>6}  {'Class':<15}" + "".join(f"  {c[:3].upper()}" for c in COMPILERS) + "  Patterns"
    print(header)
    print("-" * len(header))
    for r in bugs:
        flags = "".join(
            f"  {'  ✗' if c in r['affected_compilers'] else '   '}"
            for c in COMPILERS
        )
        pats = ",".join(p[1] for p in r.get("pattern_sequence",[]))
        print(f"{r['bug_id']:<12} {r['total_score']:>6.2f}  {r['compiler_class']:<15}{flags}  {pats}")
    print(f"\nTotal: {len(bugs)} bugs")


# ── main ──────────────────────────────────────────────────────────────────────

KNOWN_COMPILERS = [
    "onnxruntime","torchscript","torch_compile",
    "xla","tvm","tensorflow","tflite","openvino",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce Trion real compiler bugs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--bug",     metavar="ID",  help="Reproduce single bug, e.g. bug_0007")
    ap.add_argument("--class",   dest="cls",    choices=KNOWN_COMPILERS,
                    help="Reproduce all bugs for one compiler")
    ap.add_argument("--list",    action="store_true", help="List without running")
    ap.add_argument("--max",     type=int, default=50,
                    help="Max bugs to run (default 50)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print full error messages on failure")
    args = ap.parse_args()

    bugs = load_bugs(bug_filter=args.bug, class_filter=args.cls)
    if not bugs:
        sys.exit("No matching bugs found.")

    if args.list:
        list_bugs(bugs)
        return

    bugs = bugs[: args.max]

    # Summary from index
    with open(os.path.join(BUGS_DIR, "index.json")) as fp:
        index = json.load(fp)

    from collections import Counter
    cls_count: Counter = Counter()
    comp_count: Counter = Counter()
    for entry in index:
        cls_count[entry["class"]] += 1
        for c in entry["compilers"]:
            comp_count[c] += 1

    print("=" * 72)
    print("Trion Bug Reproducer — real compiler bugs only")
    print("=" * 72)
    print(f"Total unique bugs in bugs/: {len(index)}")
    print("Bugs per compiler:")
    for c in KNOWN_COMPILERS:
        if comp_count[c]:
            print(f"  {c:<16}  {comp_count[c]:3d}")
    print(f"\nReproducing {len(bugs)} bug(s) ...\n")

    confirmed = not_repro = skipped = 0
    for report in bugs:
        ok, summary = reproduce(report, verbose=args.verbose)
        print(summary)
        print()
        if "SKIP" in summary:
            skipped += 1
        elif ok:
            confirmed += 1
        else:
            not_repro += 1

    tested = confirmed + not_repro
    print("=" * 72)
    print(f"Results: {confirmed}/{tested} confirmed  |  {not_repro} not reproduced  |  {skipped} skipped")


if __name__ == "__main__":
    main()
