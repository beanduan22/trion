#!/usr/bin/env python3
"""
bug_0198 — DISCREPANCY bug
============================================================
Compiler class : openvino
Affected       : openvino
Total score    : 0.0625
Input shape    : [1, 3, 32, 32]
Patterns       : layout/space_to_depth_block -> constant/redundant_reshape -> fusion/conv_bn_relu6 -> broadcast/logsumexp_step -> fusion/conv_relu_conv_bn -> normalization/l2_normalize

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'openvino': '0.0625'}

Usage:
    python reproduce/bug_0198.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0198.onnx")

def test_discrepancy_openvino():
    """
    DISCREPANCY bug on openvino: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.0625)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
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
        print(f"  [openvino] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.0625)")
    else:
        print(f"  [openvino] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.0625)")


if __name__ == "__main__":
    print("Reproducing bug_0198 [DISCREPANCY]")
    print("  patterns : layout/space_to_depth_block -> constant/redundant_reshape -> fusion/conv_bn_relu6 -> broadcast/logsumexp_step -> fusion/conv_relu_conv_bn -> normalization/l2_normalize")
    print("  input    : [1, 3, 32, 32]")
    test_discrepancy_openvino()
