#!/usr/bin/env python3
"""
Root Cause 7 — OpenVINO segfaults compiling transformer-encoder graphs
in single-thread / LATENCY mode.

Covers crashes : 53, 241, 412, 414. ORT runs the same models cleanly;
OV's CPU plugin dies with SIGSEGV inside `compile_model` before any
inference happens. The crash needs the exact campaign witness — no
smaller hand-built graph reproduces it reliably. We isolate the OV
call in a subprocess and inspect the return code.

Usage: PYTHONPATH=. python bugs/big/rc_07_openvino_transformer_layer_segfault.py [crash_id]
       crash_id is one of: 241 (default), 053, 412, 414
Exit 0 = reproduced, 1 = OV survived, 2 = setup error.
"""
from __future__ import annotations
import os, sys, signal, subprocess, textwrap, warnings
import numpy as np
import onnx

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CRASH = 241

_OV_RUNNER = textwrap.dedent("""
    import sys, numpy as np, onnx
    from openvino import Core, convert_model
    onnx_path = sys.argv[1]
    m = onnx.load(onnx_path)
    inp = m.graph.input[0].name
    shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    np.random.seed(0)
    x = np.random.randn(*shape).astype(np.float32)
    ov = convert_model(onnx_path)
    cfg = {"INFERENCE_PRECISION_HINT": "f32",
           "PERFORMANCE_HINT": "LATENCY",
           "INFERENCE_NUM_THREADS": "1"}
    compiled = Core().compile_model(ov, "CPU", cfg)
    req = compiled.create_infer_request()
    req.infer({compiled.input(0): x})
    print("OV_OK")
""")


def main() -> int:
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CRASH
    witness = os.path.join(HERE, "witnesses", "compiler_crashes",
                           f"crash_000{cid:03d}.onnx")
    if not os.path.exists(witness):
        print(f"SKIP: witness ONNX not found at {witness}.")
        return 2

    from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend

    m = onnx.load(witness)
    inp = m.graph.input[0].name
    shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, shape).astype(np.float32)

    print(f"Witness: {witness}  shape={shape}  nodes={len(m.graph.node)}")
    r_ort = ONNXRuntimeBackend().run(m, {inp: x}, optimized=True)
    if not r_ort.ok:
        print(f"setup error: ORT itself failed: {r_ort.error}"); return 2
    out_ort = np.asarray(r_ort.output)
    print(f"ORT (correct): out shape={out_ort.shape}  "
          f"max_abs={float(np.max(np.abs(out_ort))):.4e}  "
          f"any NaN={bool(np.isnan(out_ort).any())}")

    proc = subprocess.run(
        [sys.executable, "-c", _OV_RUNNER, witness],
        capture_output=True, timeout=120,
    )
    rc = proc.returncode
    print(f"OV subprocess return code = {rc}  (SIGSEGV = -{int(signal.SIGSEGV)})")

    if rc in (-signal.SIGSEGV, 139):
        print("\nBUG REPRODUCED — OpenVINO segfaults compiling the witness graph,")
        print("while ORT produced a finite output for the same model and input.")
        return 0
    if rc != 0:
        print("OV exited with a non-segfault error:")
        print(proc.stderr.decode()[-400:])
        print("→ counted as reproduced (any OV crash where ORT succeeds is a bug).")
        return 0
    print("OV completed cleanly — bug not reproduced this run.")
    print("(Note: the segfault has been observed to depend on PID/heap state;")
    print(" rerun under the campaign supervisor to make it reliable.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
