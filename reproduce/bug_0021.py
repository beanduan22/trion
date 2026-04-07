#!/usr/bin/env python3
"""
bug_0021 — DISCREPANCY bug
============================================================
Compiler class : tflite
Affected       : openvino, tflite
Total score    : 2.8146
Input shape    : [1, 64, 128]
Patterns       : broadcast/sqrt_div_rms -> attention/self_attention_residual -> branch/residual_add_relu -> broadcast/cumsum -> branch/residual_add_relu -> constant/log_exp_cancel

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'tflite': '1.0000', 'openvino': '0.4071'}

Usage:
    python reproduce/bug_0021.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0021.onnx")

def test_discrepancy_openvino():
    """
    DISCREPANCY bug on openvino: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.4071)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 64, 128]).astype(np.float32)}
    try:
        from trion.oracle.openvino_backend import OpenVINOBackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = OpenVINOBackend(cfg)
    except Exception as e:
        print(f"  SKIP [openvino] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [openvino] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [openvino] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [openvino] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.4071)")
    else:
        print(f"  [openvino] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.4071)")

def test_discrepancy_tflite():
    """
    DISCREPANCY bug on tflite: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 64, 128]).astype(np.float32)}
    try:
        from trion.oracle.tflite_backend import TFLiteBackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = TFLiteBackend(cfg)
    except Exception as e:
        print(f"  SKIP [tflite] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [tflite] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [tflite] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [tflite] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=1.0000)")
    else:
        print(f"  [tflite] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=1.0000)")


if __name__ == "__main__":
    print("Reproducing bug_0021 [DISCREPANCY]")
    print("  patterns : broadcast/sqrt_div_rms -> attention/self_attention_residual -> branch/residual_add_relu -> broadcast/cumsum -> branch/residual_add_relu -> constant/log_exp_cancel")
    print("  input    : [1, 64, 128]")
    test_discrepancy_openvino()
    test_discrepancy_tflite()
