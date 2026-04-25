#!/usr/bin/env python3
"""
Root Cause 5 — XLA / JAX numerically drifts vs ORT on Cast(fp32->int32->fp32)
chains embedded in matmul / Conv graphs.

Covers bugs : 529, 635, 796, 928, 990  (+ 579 in torch_compile, same cluster)
Backends    : onnxruntime (reference) vs xla (jax.jit)

Each witness contains a `cp_roundtrip_int32_trunc[_neg]` /
`cast_fp32_int32_roundtrip` segment — `Add(x, +/-0.5) -> Cast(int32) ->
Cast(fp32) -> Mul(x, 1)` — followed by additional matmuls that scale
the rounding residual. ORT preserves the truncation; XLA's algebraic
simplifier almost-elides the cast pair when surrounded by enough
matmul math, leaving 1-3% rel_L2 drift even though absolute errors
stay near the noise floor of the surrounding matmul.

A 4-node hand-built `Add->Cast->Cast->Mul` does NOT reproduce —
the simplifier needs upstream MatMul context. We therefore use the
campaign witness ONNX + the recorded input the campaign caught.

Usage: PYTHONPATH=. python bugs/big/rc_05_xla_int32_cast_chain_drift.py [bug_id]
       bug_id is one of: 529 (default), 579, 635, 796, 928, 990
Exit 0 = bug reproduced, 1 = below tolerance, 2 = setup error.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import onnx

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WITNESSES = os.path.join(HERE, "witnesses")
TOL = 0.01
DEFAULT_BUG = 529


def _paths(bug_id: int):
    return (os.path.join(WITNESSES, f"bug_000{bug_id:03d}.onnx"),
            os.path.join(WITNESSES, f"bug_000{bug_id:03d}.input.npy"))


def main() -> int:
    bug_id = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUG
    onnx_path, npy_path = _paths(bug_id)
    if not (os.path.exists(onnx_path) and os.path.exists(npy_path)):
        print(f"SKIP: witness for bug {bug_id} not found under {WITNESSES}/")
        print(f"      expected {os.path.basename(onnx_path)} + {os.path.basename(npy_path)}")
        return 2

    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
    from trion.oracle.xla_backend import XLABackend

    m = onnx.load(onnx_path)
    inp = m.graph.input[0].name
    x = np.load(npy_path)

    cast_idxs = [i for i, n in enumerate(m.graph.node) if n.op_type == "Cast"]
    print(f"Witness: bug_{bug_id:03d}  shape={list(x.shape)}  nodes={len(m.graph.node)}")
    print(f"Cast nodes (roundtrip pair): {cast_idxs}")

    r_ort = ONNXRuntimeBackend().run(m, {inp: x}, optimized=True)
    r_xla = XLABackend().run(m, {inp: x}, optimized=True)
    if not r_ort.ok:
        print(f"setup error: ORT failed: {r_ort.error}"); return 2
    if not r_xla.ok:
        print(f"XLA reported error: {r_xla.error}")
        return 0

    ref = np.asarray(r_ort.output, np.float64).ravel()
    out = np.asarray(r_xla.output, np.float64).ravel()
    rel = float(np.linalg.norm(ref - out) / (np.linalg.norm(ref) + 1e-8))
    abs_max = float(np.max(np.abs(ref - out)))

    print(f"ORT  first 5 : {ref[:5]}")
    print(f"XLA  first 5 : {out[:5]}")
    print(f"rel_L2(XLA, ORT) = {rel:.6f}   max_abs_diff = {abs_max:.4e}")
    print(f"tolerance        = {TOL}")
    bad = np.argsort(np.abs(ref - out))[-3:]
    print("worst positions:")
    for i in bad:
        print(f"  [{int(i):4d}] ORT={ref[i]:+10.6f}  XLA={out[i]:+10.6f}  diff={ref[i]-out[i]:+.3e}")

    if rel > TOL or abs_max > 1e-2:
        print("\nBUG REPRODUCED — XLA disagrees with ORT past the campaign's 1% tolerance.")
        print("Root cause: int32 cast-roundtrip in a matmul/conv chain — XLA's")
        print("algebraic simplifier elides part of the truncation that ORT preserves.")
        return 0
    print("not reproduced (within tolerance)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
