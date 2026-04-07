#!/usr/bin/env python3
"""
bug_0177 — DISCREPANCY bug
============================================================
Compiler class : tensorflow
Affected       : tensorflow
Total score    : 0.2844
Input shape    : [1, 128, 16, 16]
Patterns       : normalization/batchnorm_relu6 -> normalization/ada_layer_norm -> layout/reflect_pad_conv -> fusion/conv_bn_relu6 -> branch/channel_gating_branch -> broadcast/where_mask

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'tensorflow': '0.2844'}

Usage:
    python reproduce/bug_0177.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0177.onnx")

def test_discrepancy_tensorflow():
    """
    DISCREPANCY bug on tensorflow: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.2844)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 16, 16]).astype(np.float32)}
    try:
        from trion.oracle.tf_backend import TFBackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = TFBackend(cfg)
    except Exception as e:
        print(f"  SKIP [tensorflow] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [tensorflow] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [tensorflow] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [tensorflow] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.2844)")
    else:
        print(f"  [tensorflow] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.2844)")


if __name__ == "__main__":
    print("Reproducing bug_0177 [DISCREPANCY]")
    print("  patterns : normalization/batchnorm_relu6 -> normalization/ada_layer_norm -> layout/reflect_pad_conv -> fusion/conv_bn_relu6 -> branch/channel_gating_branch -> broadcast/where_mask")
    print("  input    : [1, 128, 16, 16]")
    test_discrepancy_tensorflow()
