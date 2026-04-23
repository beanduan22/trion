"""Shared helpers for root-cause minimal repros.

Each repro builds a small ONNX graph and runs it through the two clusters:
    reference cluster : onnxruntime (CPU) + pytorch_eager (onnx2torch)
    suspect cluster   : whichever backend(s) the cluster identifies

Run any repro from the `trion/` repo root:

    python bug_root_cause_catalog/repros/unique_NNNN_min.py

Exit 0 = divergence reproduced; 1 = backends agreed (bug gone); 2 = setup error.
"""
from __future__ import annotations

import os
import sys
import numpy as np


def locate_repo_root():
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(here, "xcomp")):
            if here not in sys.path:
                sys.path.insert(0, here)
            return here
        here = os.path.dirname(here)
    return None


def run_backend(name, model_proto, inputs):
    """Call an xcomp oracle backend. Returns (output, err_string)."""
    import importlib
    spec = {
        "pytorch_eager":  ("xcomp.oracle.pytorch_backend",       "PyTorchEagerBackend"),
        "onnxruntime":    ("xcomp.oracle.onnxruntime_backend",   "OnnxRuntimeBackend"),
        "torchscript":    ("xcomp.oracle.torchscript_backend",   "TorchScriptBackend"),
        "torch_compile":  ("xcomp.oracle.torch_compile_backend", "TorchCompileBackend"),
        "xla":            ("xcomp.oracle.xla_backend",           "XLABackend"),
        "tvm":            ("xcomp.oracle.tvm_backend",           "TVMBackend"),
        "openvino":       ("xcomp.oracle.openvino_backend",      "OpenVINOBackend"),
        "tensorflow":     ("xcomp.oracle.tf_backend",            "TFBackend"),
    }
    mod_name, cls_name = spec[name]
    try:
        cls = getattr(importlib.import_module(mod_name), cls_name)
        b = cls()
        out = b.run(model_proto, inputs)
        return out, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def rel_l2(a, b):
    if a is None or b is None:
        return float("inf")
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))


def report(expected_pair, suspect_pair, outputs, tolerance):
    """Print rel_l2 numbers and return (exit_code, worst_rel_l2)."""
    ref_name, ref = expected_pair
    rows = []
    for name, val in suspect_pair:
        diff = rel_l2(val, ref)
        rows.append((name, diff))
    print(f"\nreference: {ref_name}")
    print(f"tolerance: {tolerance}")
    for name, diff in rows:
        flag = "BUG" if diff > tolerance else "ok"
        print(f"  rel_l2({name:<14} vs {ref_name:<14}) = {diff:.6f}   [{flag}]")
    worst = max((d for _, d in rows), default=0.0)
    return (0 if worst > tolerance else 1), worst
