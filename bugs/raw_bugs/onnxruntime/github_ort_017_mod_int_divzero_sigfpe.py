#!/usr/bin/env python3
"""
Bug ID     : github_ort_017
Source     : Trion campaign v5 compat check (2026-04-16)
Compiler   : ONNXRuntime 1.24.4 (CPU)
Patterns   : Mod(fmod=0, int32) with zero divisor
Root cause : ORT's C++ kernel for integer Mod executes raw `a % b` without
             guarding `b == 0`.  On x86 this triggers a hardware SIGFPE
             (Floating Point Exception, signal 8) that kills the process.
             The ONNX spec says Mod behaviour on zero divisor is
             implementation-defined, but a process-killing signal is
             unacceptable — ORT should either raise a RuntimeError or
             return 0 / produce a defined result.
             Float Mod with fmod=1 handles zero divisor correctly (returns
             NaN), so only the integer path is affected.
Tolerance  : N/A (crash bug, not numerical divergence)

Exit 0 = BUG REPRODUCED (SIGFPE on the subprocess)
Exit 1 = not reproduced (ORT handled zero divisor gracefully)
Exit 2 = missing deps
"""
import subprocess
import sys
import textwrap

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not installed")
    sys.exit(2)

ort_version = onnxruntime.__version__

CHILD_SCRIPT = textwrap.dedent(r"""
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

nodes = [helper.make_node("Mod", ["A", "B"], ["Y"], fmod=0)]
graph = helper.make_graph(
    nodes, "g",
    [helper.make_tensor_value_info("A", TensorProto.INT32, [2]),
     helper.make_tensor_value_info("B", TensorProto.INT32, [2])],
    [helper.make_tensor_value_info("Y", TensorProto.INT32, [2])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CPUExecutionProvider"],
)
a = np.array([7, 13], dtype=np.int32)
b = np.array([0, 0], dtype=np.int32)
out = sess.run(None, {"A": a, "B": b})[0]
print(f"output={out}")
""")

proc = subprocess.run(
    [sys.executable, "-c", CHILD_SCRIPT],
    capture_output=True, text=True, timeout=15,
)

print(f"OnnxRuntime version: {ort_version}")
print(f"Child exit code: {proc.returncode}")

if proc.returncode < 0:
    import signal
    sig = -proc.returncode
    sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"
    print(f"Child killed by {sig_name} (signal {sig})")
    if sig == signal.SIGFPE:
        print("BUG REPRODUCED — ORT Mod(fmod=0, int32) with B=0 triggers SIGFPE")
        sys.exit(0)
    else:
        print(f"Unexpected signal {sig_name}")
        sys.exit(1)
elif proc.returncode != 0:
    stderr = proc.stderr.strip()[:200]
    if "RUNTIME_EXCEPTION" in stderr or "division by zero" in stderr.lower():
        print(f"ORT raised exception (not SIGFPE): {stderr}")
        print("BUG REPRODUCED — ORT crashes on Mod(fmod=0, int32) with B=0")
        sys.exit(0)
    print(f"Child failed with code {proc.returncode}: {stderr}")
    sys.exit(1)
else:
    stdout = proc.stdout.strip()
    print(f"Child output: {stdout}")
    print("NOT REPRODUCED — ORT handled zero divisor gracefully")
    sys.exit(1)
