#!/usr/bin/env python3
"""
Root Cause 8 — small numerical drift cluster.

Covers bugs : 210 (TVM), 214 (XLA), 224 (TC), 297 (XLA), 392 (TS),
              413 (TS), 585 (TVM, near-zero ref -> huge rel_L2).

These are not a single root cause: each witness has matmul / LN / softmax
chains where fp32 accumulation order, LN variance recipe, or softmax
polynomial differ between ORT and the suspect backend just enough to
exceed the 1% relative tolerance. Hand-built minimal graphs do not
amplify the drift the way the witness graphs do, so this reproducer
loads the campaign's ONNX + recorded input.

Usage: PYTHONPATH=. python bugs/big/rc_08_small_numerical_drift_cluster.py
Exit 0 = >=1 bug reproduced, 1 = none did, 2 = setup error.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import onnx

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WITNESSES = os.path.join(HERE, "witnesses")
TOL = 0.01
CASES = [
    # (bug_id, suspect-backend-class, fuzzer-rel_L2 from campaign)
    (210, "tvm",            0.014),
    (214, "xla",            0.045),
    (224, "torch_compile",  0.015),
    (297, "xla",            0.010),
    (392, "torchscript",    0.034),
    (413, "torchscript",    0.018),
    (585, "tvm",            145303.0),
]


def _backend(name: str):
    if name == "tvm":
        from trion.oracle.tvm_backend import TVMBackend; return TVMBackend()
    if name == "xla":
        from trion.oracle.xla_backend import XLABackend; return XLABackend()
    if name == "torch_compile":
        from trion.oracle.torch_compile_backend import TorchCompileBackend; return TorchCompileBackend()
    if name == "torchscript":
        from trion.oracle.torchscript_backend import TorchScriptBackend; return TorchScriptBackend()
    raise ValueError(name)


def main() -> int:
    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
    ort = ONNXRuntimeBackend()

    print(f"{'bug':>4} {'backend':>14} {'campaign':>10} {'this run':>11} {'max_abs':>10} verdict")
    n_repro = n_skip = 0
    for bid, bname, fuzz_rel in CASES:
        onnx_path = os.path.join(WITNESSES, f"bug_000{bid:03d}.onnx")
        npy_path  = os.path.join(WITNESSES, f"bug_000{bid:03d}.input.npy")
        if not (os.path.exists(onnx_path) and os.path.exists(npy_path)):
            print(f"{bid:>4} {bname:>14} {'-':>10} {'-':>11} {'-':>10} skip (no witness)")
            n_skip += 1; continue
        m = onnx.load(onnx_path)
        inp_name = m.graph.input[0].name
        x = np.load(npy_path)
        be = _backend(bname)
        if not be.is_available():
            print(f"{bid:>4} {bname:>14} {'-':>10} {'-':>11} {'-':>10} skip ({bname} unavailable)")
            n_skip += 1; continue
        r1 = ort.run(m, {inp_name: x}, optimized=True)
        r2 = be.run(m, {inp_name: x}, optimized=True)
        if not (r1.ok and r2.ok):
            err = r1.error if not r1.ok else r2.error
            print(f"{bid:>4} {bname:>14} {fuzz_rel:>10.4f} {'crash':>11} {'-':>10} crash: {(err or '')[:30]}")
            n_repro += 1; continue
        ref = np.asarray(r1.output, np.float64).ravel()
        out = np.asarray(r2.output, np.float64).ravel()
        rel = float(np.linalg.norm(ref - out) / (np.linalg.norm(ref) + 1e-8))
        ad  = float(np.max(np.abs(ref - out)))
        verdict = "REPRODUCED" if rel > TOL else "within tolerance"
        if rel > TOL: n_repro += 1
        print(f"{bid:>4} {bname:>14} {fuzz_rel:>10.4f} {rel:>11.6f} {ad:>10.3e} {verdict}")

    print()
    print(f"Reproduced: {n_repro}/{len(CASES) - n_skip} (with {n_skip} skipped)")
    print("\nThese bugs are small numerical drifts that exceed the campaign's 1%")
    print("relative tolerance; absolute differences are typically <0.1, often")
    print("near the fp32 noise floor of the surrounding matmul/LN math.")
    return 0 if n_repro > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
