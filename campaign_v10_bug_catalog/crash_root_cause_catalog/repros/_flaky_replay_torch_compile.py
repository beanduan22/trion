#!/usr/bin/env python3
"""
Clean-process torch.compile replay for ONE crash model. Uses the same
xcomp TorchCompileBackend the campaign used.

Usage: python _flaky_replay_torch_compile.py <path-to-crash_NNNNNN.onnx> <gpu_id>
"""
import os
import sys

if len(sys.argv) < 2:
    print("missing arg"); sys.exit(3)

onnx_path = sys.argv[1]
gpu_id = sys.argv[2] if len(sys.argv) > 2 else "1"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import onnx

# Locate repo root for xcomp import
here = os.path.abspath(os.path.dirname(__file__))
for _ in range(6):
    if os.path.isdir(os.path.join(here, "xcomp")):
        sys.path.insert(0, here); break
    here = os.path.dirname(here)

try:
    from xcomp.oracle.torch_compile_backend import TorchCompileBackend
except Exception as e:
    print(f"[setup] xcomp backend unavailable: {e}"); sys.exit(3)

backend = TorchCompileBackend()
model = onnx.load(onnx_path)
inp = model.graph.input[0]
shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
np.random.seed(0)
inputs = {inp.name: np.random.randn(*shape).astype(np.float32)}

result = backend.run(model, inputs, optimized=True)
if result.ok:
    print(f"RAN OK: out.shape={result.output.shape}")
    sys.exit(1)

msg = result.error or ""
if "CUDNN_STATUS_INTERNAL_ERROR" in msg:
    print(f"CRASH REPRO (cudnn-internal): {msg.splitlines()[-1][:180]}")
    sys.exit(0)
if "out of memory" in msg.lower():
    print(f"ENV OOM: {msg.splitlines()[-1][:180]}")
    sys.exit(2)
print(f"DIFFERENT ERROR: {msg.splitlines()[-1][:220]}")
sys.exit(2)
