#!/usr/bin/env python3
"""
bug_0135 — DISCREPANCY bug
============================================================
Compiler class : xla
Affected       : tvm, xla
Total score    : 5.0000
Input shape    : [1, 16, 128]
Patterns       : broadcast/mul_add_relu -> layout/squeeze_unsqueeze -> attention/multi_head_self_attention -> branch/glu -> broadcast/where_mask -> layout/squeeze_unsqueeze

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'xla': '1.0000', 'tvm': '1.0000'}

Usage:
    python reproduce/bug_0135.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0135.onnx")

def test_discrepancy_tvm():
    """
    DISCREPANCY bug on tvm: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 16, 128]).astype(np.float32)}
    try:
        from trion.oracle.tvm_backend import TVMBackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = TVMBackend(cfg)
    except Exception as e:
        print(f"  SKIP [tvm] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [tvm] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [tvm] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [tvm] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=1.0000)")
    else:
        print(f"  [tvm] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=1.0000)")

def test_discrepancy_xla():
    """
    DISCREPANCY bug on xla: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 16, 128]).astype(np.float32)}
    try:
        from trion.oracle.xla_backend import XLABackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = XLABackend(cfg)
    except Exception as e:
        print(f"  SKIP [xla] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [xla] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [xla] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [xla] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=1.0000)")
    else:
        print(f"  [xla] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=1.0000)")


if __name__ == "__main__":
    print("Reproducing bug_0135 [DISCREPANCY]")
    print("  patterns : broadcast/mul_add_relu -> layout/squeeze_unsqueeze -> attention/multi_head_self_attention -> branch/glu -> broadcast/where_mask -> layout/squeeze_unsqueeze")
    print("  input    : [1, 16, 128]")
    test_discrepancy_tvm()
    test_discrepancy_xla()
