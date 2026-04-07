#!/usr/bin/env python3
"""
bug_0067 — DISCREPANCY bug
============================================================
Compiler class : torch_compile
Affected       : torch_compile
Total score    : 0.8182
Input shape    : [1, 32, 32, 32]
Patterns       : broadcast/reciprocal_mul -> branch/spatial_attention_cbam -> constant/slice_full_range -> broadcast/euclidean_norm_broadcast -> fusion/conv_gelu -> layout/depth_to_space_block

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'torch_compile': '0.3999'}

Usage:
    python reproduce/bug_0067.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0067.onnx")

def test_discrepancy_torch_compile():
    """
    DISCREPANCY bug on torch_compile: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.3999)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 32, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.torch_compile_backend import TorchCompileBackend
        
        be = TorchCompileBackend()
    except Exception as e:
        print(f"  SKIP [torch_compile] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [torch_compile] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [torch_compile] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [torch_compile] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.3999)")
    else:
        print(f"  [torch_compile] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.3999)")


if __name__ == "__main__":
    print("Reproducing bug_0067 [DISCREPANCY]")
    print("  patterns : broadcast/reciprocal_mul -> branch/spatial_attention_cbam -> constant/slice_full_range -> broadcast/euclidean_norm_broadcast -> fusion/conv_gelu -> layout/depth_to_space_block")
    print("  input    : [1, 32, 32, 32]")
    test_discrepancy_torch_compile()
