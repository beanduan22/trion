#!/usr/bin/env python3
"""
Bug v2-0010 — OnnxRuntime wrong output vs ONNX spec reference.
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT Resize nearest_ceil combined with pow+conv sequence — optimizer reorders the pow before resize changing which source pixel gets raised to the power.
Report to : https://github.com/microsoft/onnxruntime/issues
           (label: bug, regression, Resize / TopK / CumSum etc.)

Oracle: ORT_ENABLE_ALL output vs pytorch_eager (onnx2torch) reference.
        ORT_DISABLE_ALL gives the same wrong answer — this is a fundamental
        implementation divergence, not an optimization regression.

The ONNX model is embedded in unique_2629.py (the reference repro).
Run: python v2_0010_onnxruntime.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os, subprocess

_ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'unique_2629.py')

if not os.path.exists(_ref):
    print("Reference unique_2629.py not found")
    sys.exit(1)

r = subprocess.run([sys.executable, _ref], capture_output=True, timeout=120)
sys.stdout.buffer.write(r.stdout)
sys.stderr.buffer.write(r.stderr)
if r.returncode == 0 and b'BUG REPRODUCED' in r.stdout:
    sys.exit(0)
sys.exit(r.returncode)
