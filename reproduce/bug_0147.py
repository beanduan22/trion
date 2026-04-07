#!/usr/bin/env python3
"""
bug_0147 — DISCREPANCY bug
============================================================
Compiler class : onnxruntime
Affected       : onnxruntime
Total score    : 2.0000
Input shape    : [1, 64, 16, 16]
Patterns       : normalization/instancenorm_relu -> fusion/depthwise_conv_bn_relu -> layout/channel_shuffle -> fusion/conv_bn_relu

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'onnxruntime': '1.0000'}

Usage:
    python reproduce/bug_0147.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0147.onnx")

def test_discrepancy_onnxruntime():
    """
    DISCREPANCY bug on onnxruntime: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 64, 16, 16]).astype(np.float32)}
    try:
        from trion.oracle.onnxruntime_backend import ONNXRuntimeBackend
        
        be = ONNXRuntimeBackend()
    except Exception as e:
        print(f"  SKIP [onnxruntime] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [onnxruntime] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [onnxruntime] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [onnxruntime] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=1.0000)")
    else:
        print(f"  [onnxruntime] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=1.0000)")


if __name__ == "__main__":
    print("Reproducing bug_0147 [DISCREPANCY]")
    print("  patterns : normalization/instancenorm_relu -> fusion/depthwise_conv_bn_relu -> layout/channel_shuffle -> fusion/conv_bn_relu")
    print("  input    : [1, 64, 16, 16]")
    test_discrepancy_onnxruntime()
