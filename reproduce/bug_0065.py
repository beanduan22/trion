#!/usr/bin/env python3
"""
bug_0065 — DISCREPANCY bug
============================================================
Compiler class : torch_compile
Affected       : torch_compile
Total score    : 0.4107
Input shape    : [1, 16, 256]
Patterns       : attention/gated_mlp_block -> broadcast/swish -> constant/div_by_constant -> attention/gated_mlp_block -> constant/cast_roundtrip -> broadcast/sub_mul_add

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'torch_compile': '0.1890'}

Usage:
    python reproduce/bug_0065.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0065.onnx")

def test_discrepancy_torch_compile():
    """
    DISCREPANCY bug on torch_compile: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.1890)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 16, 256]).astype(np.float32)}
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
        print(f"  [torch_compile] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.1890)")
    else:
        print(f"  [torch_compile] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.1890)")


if __name__ == "__main__":
    print("Reproducing bug_0065 [DISCREPANCY]")
    print("  patterns : attention/gated_mlp_block -> broadcast/swish -> constant/div_by_constant -> attention/gated_mlp_block -> constant/cast_roundtrip -> broadcast/sub_mul_add")
    print("  input    : [1, 16, 256]")
    test_discrepancy_torch_compile()
