#!/usr/bin/env python3
"""
Trion Bug Reproducer — v2
=========================
Reproduces all verified compiler bugs and generates bug-report-ready output.

Two bug types are handled separately:
  CRASH      — backend raises an exception (crashes[] list is non-empty)
  DISCREPANCY — optimization changes output by more than threshold
               (delta_opt[compiler] > DISC_THRESH, crashes[] is empty)

Usage:
    # Reproduce every bug
    python reproduce_bugs.py

    # Reproduce one compiler class
    python reproduce_bugs.py --class torch_compile
    python reproduce_bugs.py --class onnxruntime
    python reproduce_bugs.py --class tflite
    python reproduce_bugs.py --class openvino
    python reproduce_bugs.py --class tensorflow
    python reproduce_bugs.py --class tvm
    python reproduce_bugs.py --class xla
    python reproduce_bugs.py --class torchscript

    # Reproduce a single bug
    python reproduce_bugs.py --bug bug_0071

    # List all bugs without running
    python reproduce_bugs.py --list

    # Show only crash bugs or discrepancy bugs
    python reproduce_bugs.py --type crash
    python reproduce_bugs.py --type discrepancy

    # Generate bug-report-ready output (verbose errors + patterns)
    python reproduce_bugs.py --report

    # Limit number of bugs to test
    python reproduce_bugs.py --max 20

    # Verbose: print full error on failure
    python reproduce_bugs.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx

BUGS_DIR    = os.path.join(os.path.dirname(__file__), "bugs")
DISC_THRESH = 0.05   # relative-difference threshold — same as oracle

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

KNOWN_COMPILERS = [
    "onnxruntime", "torchscript", "torch_compile",
    "xla", "tvm", "tensorflow", "tflite", "openvino",
]

# ── Backend loader ─────────────────────────────────────────────────────────────

def _load_backend(name: str):
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    a, b = a.flatten()[:n].astype(np.float32), b.flatten()[:n].astype(np.float32)
    diff = float(np.linalg.norm(a - b))
    norm = float(max(np.linalg.norm(b), 1e-8))
    return diff / norm


def make_inputs(model: onnx.ModelProto, shape: list) -> Dict[str, np.ndarray]:
    inp = model.graph.input[0]
    name = inp.name
    rng  = np.random.default_rng(42)
    return {name: rng.standard_normal(shape).astype(np.float32)}


def _fmt_patterns(seq: list) -> str:
    return " → ".join(f"{cat}/{pat}" for cat, pat in seq)


# ── Bug loading ────────────────────────────────────────────────────────────────

def load_bugs(
    bug_filter:   Optional[str] = None,
    class_filter: Optional[str] = None,
    type_filter:  Optional[str] = None,   # "crash" | "discrepancy"
) -> List[dict]:
    index_path = os.path.join(BUGS_DIR, "index.json")
    if not os.path.exists(index_path):
        sys.exit(f"Bug index not found: {index_path}")

    with open(index_path) as fp:
        index = json.load(fp)

    results = []
    for entry in index:
        bid = entry["bug_id"]
        if bug_filter and bid != bug_filter:
            continue
        if class_filter and entry["class"] != class_filter:
            continue
        report_path = os.path.join(BUGS_DIR, f"{bid}_report.json")
        onnx_path   = os.path.join(BUGS_DIR, f"{bid}.onnx")
        if not os.path.exists(report_path) or not os.path.exists(onnx_path):
            continue
        with open(report_path) as fp:
            report = json.load(fp)
        report["_onnx"] = onnx_path

        # Determine bug type
        has_crash = bool(report.get("crashes", []))
        has_disc  = any(v > DISC_THRESH for v in report.get("delta_opt", {}).values())
        btype = "crash" if has_crash and not has_disc else \
                "discrepancy" if has_disc and not has_crash else \
                "crash+discrepancy" if has_crash and has_disc else "unknown"
        report["_type"] = btype

        if type_filter:
            if type_filter == "crash"       and "crash"       not in btype: continue
            if type_filter == "discrepancy" and "discrepancy" not in btype: continue

        results.append(report)

    return results


# ── Reproduce: CRASH bug ───────────────────────────────────────────────────────

def reproduce_crash(report: dict, verbose: bool, report_mode: bool) -> Tuple[str, bool]:
    """Verify that the crash is still present."""
    bid        = report["bug_id"]
    onnx_path  = report["_onnx"]
    ishape     = report.get("input_shape", [1, 3, 32, 32])
    compilers  = list({
        c.split("+")[0].split("-")[0]
        for c in report.get("crashes", [])
        if c not in ("pytorch_eager", "ref")
    })
    saved_errs = report.get("errors", {})
    patterns   = report.get("pattern_sequence", [])

    model  = onnx.load(onnx_path)
    inputs = make_inputs(model, ishape)

    lines = [
        f"{'─'*72}",
        f"Bug {bid}  type=CRASH  class={report.get('compiler_class','?')}  score={report['total_score']:.4f}",
        f"  patterns : {_fmt_patterns(patterns)}",
        f"  input    : {ishape}",
        f"  compilers: {compilers}",
    ]

    if report_mode:
        lines.append(f"\n  [BUG REPORT INFO]")
        lines.append(f"  Reproducible ONNX: {onnx_path}")
        for c, msg in saved_errs.items():
            if msg:
                lines.append(f"  Error [{c}]: {msg[:300]}")

    confirmed = False

    for compiler in compilers:
        try:
            be = _load_backend(compiler)
        except Exception as exc:
            lines.append(f"  [{compiler}] SKIP (not installed): {exc}")
            continue

        for opt in [True, False]:
            tag = f"{compiler}+{'opt' if opt else 'noopt'}"
            try:
                res = be.run(model, inputs, optimized=opt)
            except Exception as exc:
                lines.append(f"  [{tag}] EXCEPTION: {str(exc)[:200]}")
                confirmed = True
                continue

            if res.crashed:
                err = (res.error or "")[:200]
                lines.append(f"  [{tag}] ✗ CRASH confirmed: {err}")
                if verbose and res.error and res.error != err:
                    lines.append(f"    full:\n{res.error}")
                confirmed = True
            elif res.output is None:
                lines.append(f"  [{tag}] ✗ NO OUTPUT")
                confirmed = True
            else:
                lines.append(f"  [{tag}] ✓ no crash (may be fixed)")

    status = "CONFIRMED" if confirmed else "NOT REPRODUCED (may be fixed)"
    lines.append(f"\n  → {status}")
    return "\n".join(lines), confirmed


# ── Reproduce: DISCREPANCY bug ─────────────────────────────────────────────────

def reproduce_discrepancy(report: dict, verbose: bool, report_mode: bool) -> Tuple[str, bool]:
    """
    Verify optimization-induced discrepancy: run same backend with opt=True
    and opt=False, check that rel_diff(opt_out, noopt_out) > DISC_THRESH.
    """
    bid       = report["bug_id"]
    onnx_path = report["_onnx"]
    ishape    = report.get("input_shape", [1, 3, 32, 32])
    delta_opt = report.get("delta_opt", {})
    patterns  = report.get("pattern_sequence", [])

    # Compilers that show discrepancy
    disc_compilers = [c for c, v in delta_opt.items() if v > DISC_THRESH
                      and c not in ("pytorch_eager",)]

    model  = onnx.load(onnx_path)
    inputs = make_inputs(model, ishape)

    lines = [
        f"{'─'*72}",
        f"Bug {bid}  type=DISCREPANCY  class={report.get('compiler_class','?')}  score={report['total_score']:.4f}",
        f"  patterns     : {_fmt_patterns(patterns)}",
        f"  input        : {ishape}",
        f"  compilers    : {disc_compilers}",
        f"  expected Δopt: { {c: f'{v:.4f}' for c,v in delta_opt.items() if v > DISC_THRESH} }",
    ]

    if report_mode:
        lines.append(f"\n  [BUG REPORT INFO]")
        lines.append(f"  Reproducible ONNX: {onnx_path}")
        lines.append(f"  Bug type: optimization-induced numerical discrepancy")
        lines.append(f"  Expected: opt output ≈ noopt output (rel_diff < {DISC_THRESH})")
        lines.append(f"  Observed: rel_diff > {DISC_THRESH} after optimization")

    confirmed = False

    for compiler in disc_compilers:
        try:
            be = _load_backend(compiler)
        except Exception as exc:
            lines.append(f"  [{compiler}] SKIP (not installed): {exc}")
            continue

        try:
            res_opt   = be.run(model, inputs, optimized=True)
            res_noopt = be.run(model, inputs, optimized=False)
        except Exception as exc:
            lines.append(f"  [{compiler}] EXCEPTION during run: {str(exc)[:200]}")
            continue

        if res_opt.crashed or res_noopt.crashed:
            err = (res_opt.error or res_noopt.error or "")[:150]
            lines.append(f"  [{compiler}] CRASH during run (unexpected for discrepancy bug): {err}")
            continue

        if res_opt.output is None or res_noopt.output is None:
            lines.append(f"  [{compiler}] NO OUTPUT")
            continue

        rd = rel_diff(res_opt.output, res_noopt.output)
        expected = delta_opt.get(compiler, 0)

        if rd > DISC_THRESH:
            lines.append(
                f"  [{compiler}] ✗ DISCREPANCY confirmed  "
                f"rel_diff={rd:.4f}  (stored={expected:.4f})"
            )
            confirmed = True
        else:
            lines.append(
                f"  [{compiler}] ✓ no discrepancy  "
                f"rel_diff={rd:.4f}  (stored={expected:.4f})  "
                f"{'→ may be fixed' if expected > DISC_THRESH else ''}"
            )

    status = "CONFIRMED" if confirmed else "NOT REPRODUCED (may be fixed or input-sensitive)"
    lines.append(f"\n  → {status}")
    return "\n".join(lines), confirmed


# ── List bugs ──────────────────────────────────────────────────────────────────

def list_bugs(bugs: List[dict]) -> None:
    type_sym = {"crash": "💥", "discrepancy": "∿", "crash+discrepancy": "💥∿", "unknown": "?"}
    print(f"\n{'Bug ID':<12} {'Score':>6}  {'Type':<22} {'Class':<16}  Compilers")
    print("─" * 90)
    for r in bugs:
        btype    = r.get("_type", "?")
        sym      = type_sym.get(btype, "?")
        compilers = ",".join(r.get("affected_compilers", []))
        print(
            f"{r['bug_id']:<12} {r['total_score']:>6.2f}  "
            f"{sym} {btype:<20} {r.get('compiler_class','?'):<16}  {compilers}"
        )
    print(f"\nTotal: {len(bugs)} bugs")

    # Summary stats
    tc = Counter(r.get("_type","?") for r in bugs)
    cc = Counter()
    for r in bugs:
        for c in r.get("affected_compilers", []):
            cc[c] += 1
    print("\nBy type:")
    for t, n in tc.most_common():
        print(f"  {t:<22}  {n}")
    print("\nBy compiler:")
    for c, n in cc.most_common():
        print(f"  {c:<20}  {n}")


# ── Generate bug report summary ────────────────────────────────────────────────

def print_bug_report_summary(bugs: List[dict], results: List[Tuple[dict, bool, str]]) -> None:
    """Print a structured summary suitable for filing bug reports."""
    confirmed = [r for r in results if r[1]]
    not_repro = [r for r in results if not r[1]]

    print("\n" + "=" * 72)
    print("BUG REPORT SUMMARY")
    print("=" * 72)
    print(f"Confirmed: {len(confirmed)}  |  Not reproduced: {len(not_repro)}")

    # Group by primary compiler and bug type
    by_compiler: Dict[str, List[dict]] = defaultdict(list)
    for report, ok, _ in results:
        if ok:
            cls = report.get("compiler_class", "unknown")
            by_compiler[cls].append(report)

    for compiler, reps in sorted(by_compiler.items()):
        print(f"\n── {compiler.upper()} ({len(reps)} bugs) ──────────────────────")
        crash_bugs = [r for r in reps if "crash" in r.get("_type","")]
        disc_bugs  = [r for r in reps if "discrepancy" in r.get("_type","") and "crash" not in r.get("_type","")]
        if crash_bugs:
            print(f"  CRASH bugs ({len(crash_bugs)}):")
            for r in crash_bugs:
                errs = r.get("errors", {})
                first_err = next(iter(errs.values()), "")[:120] if errs else ""
                print(f"    {r['bug_id']:<12}  {first_err}")
        if disc_bugs:
            print(f"  DISCREPANCY bugs ({len(disc_bugs)}):")
            delta = r.get("delta_opt", {})
            for r in disc_bugs:
                delta = {c: f"{v:.3f}" for c, v in r.get("delta_opt", {}).items() if v > DISC_THRESH}
                pats  = [p[1] for p in r.get("pattern_sequence", [])]
                print(f"    {r['bug_id']:<12}  Δopt={delta}  patterns={pats}")

    print("\n── FILING TARGETS ────────────────────────────────────────────────")
    filing_map = {
        "torch_compile":  "https://github.com/pytorch/pytorch/issues",
        "torchscript":    "https://github.com/pytorch/pytorch/issues",
        "onnxruntime":    "https://github.com/microsoft/onnxruntime/issues",
        "tflite":         "https://github.com/tensorflow/tensorflow/issues",
        "tensorflow":     "https://github.com/tensorflow/tensorflow/issues",
        "xla":            "https://github.com/openxla/xla/issues",
        "tvm":            "https://github.com/apache/tvm/issues",
        "openvino":       "https://github.com/openvinotoolkit/openvino/issues",
    }
    for compiler in sorted(by_compiler):
        n   = len(by_compiler[compiler])
        url = filing_map.get(compiler, "unknown")
        print(f"  {compiler:<20}  {n} bug(s)  →  {url}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce and verify Trion compiler bugs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--bug",     metavar="ID",  help="Reproduce single bug, e.g. bug_0071")
    ap.add_argument("--class",   dest="cls",    choices=KNOWN_COMPILERS,
                    help="Filter by compiler class")
    ap.add_argument("--type",    dest="btype",  choices=["crash", "discrepancy"],
                    help="Filter by bug type")
    ap.add_argument("--list",    action="store_true", help="List bugs without running")
    ap.add_argument("--max",     type=int,  default=200, help="Max bugs to run (default 200)")
    ap.add_argument("--report",  action="store_true",
                    help="Generate bug-report-ready output (ONNX path + errors)")
    ap.add_argument("--verbose", action="store_true", help="Print full error messages")
    args = ap.parse_args()

    bugs = load_bugs(
        bug_filter   = args.bug,
        class_filter = args.cls,
        type_filter  = args.btype,
    )
    if not bugs:
        sys.exit("No matching bugs found.")

    if args.list:
        list_bugs(bugs)
        return

    bugs = bugs[: args.max]

    # Load index for header
    with open(os.path.join(BUGS_DIR, "index.json")) as fp:
        index = json.load(fp)

    cc = Counter()
    tc = Counter()
    for entry in index:
        tc[entry.get("class", "?")] += 1
        for c in entry.get("compilers", []):
            cc[c] += 1

    print("=" * 72)
    print("Trion Bug Reproducer — verified compiler bugs")
    print("=" * 72)
    print(f"Total unique bugs in index: {len(index)}")
    print("Bugs per compiler:")
    for c in KNOWN_COMPILERS:
        if cc[c]:
            print(f"  {c:<20}  {cc[c]:3d}")
    print(f"\nReproducing {len(bugs)} bug(s) ...\n")

    confirmed_n = not_repro_n = skipped_n = 0
    results: List[Tuple[dict, bool, str]] = []

    for report in bugs:
        btype = report.get("_type", "unknown")

        if btype == "crash":
            summary, ok = reproduce_crash(report, args.verbose, args.report)
        elif btype == "discrepancy":
            summary, ok = reproduce_discrepancy(report, args.verbose, args.report)
        elif btype == "crash+discrepancy":
            # Verify both: crash and discrepancy
            s1, ok1 = reproduce_crash(report, args.verbose, args.report)
            s2, ok2 = reproduce_discrepancy(report, args.verbose, args.report)
            summary  = s1 + "\n" + s2
            ok       = ok1 or ok2
        else:
            summary = f"  [{report['bug_id']}] SKIP — unknown bug type"
            ok      = False

        print(summary)
        print()
        results.append((report, ok, summary))

        if "SKIP" in summary:
            skipped_n += 1
        elif ok:
            confirmed_n += 1
        else:
            not_repro_n += 1

    tested = confirmed_n + not_repro_n
    print("=" * 72)
    print(
        f"Results: {confirmed_n}/{tested} confirmed  |  "
        f"{not_repro_n} not reproduced  |  {skipped_n} skipped"
    )

    if args.report:
        print_bug_report_summary(bugs, results)


if __name__ == "__main__":
    main()
