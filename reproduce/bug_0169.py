#!/usr/bin/env python3
"""
bug_0169 — DISCREPANCY bug
============================================================
Compiler class : tflite
Affected       : tflite
Total score    : 0.0563
Input shape    : [1, 16, 64]
Patterns       : layout/squeeze_unsqueeze -> constant/learned_temperature_scale -> attention/multi_query_attention -> broadcast/where_mask -> attention/self_attention_residual -> attention/self_attention_residual

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'tflite': '0.0563'}

Usage:
    python reproduce/bug_0169.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0169.onnx")

def test_discrepancy_tflite():
    """
    DISCREPANCY bug on tflite: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.0563)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 16, 64]).astype(np.float32)}
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
        print(f"  [tflite] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.0563)")
    else:
        print(f"  [tflite] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.0563)")


if __name__ == "__main__":
    print("Reproducing bug_0169 [DISCREPANCY]")
    print("  patterns : layout/squeeze_unsqueeze -> constant/learned_temperature_scale -> attention/multi_query_attention -> broadcast/where_mask -> attention/self_attention_residual -> attention/self_attention_residual")
    print("  input    : [1, 16, 64]")
    test_discrepancy_tflite()
