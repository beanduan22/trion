#!/usr/bin/env python3
"""
bug_0104 — DISCREPANCY bug
============================================================
Compiler class : xla
Affected       : onnxruntime, torchscript, tvm, xla
Total score    : 1.7839
Input shape    : [1, 128, 32, 32]
Patterns       : branch/add_layernorm -> fusion/conv_bn_leakyrelu -> normalization/ada_layer_norm -> layout/double_transpose -> broadcast/where_mask -> fusion/grouped_conv_bn_relu

Bug type: DISCREPANCY

Discrepancy (rel_diff opt vs noopt): {'onnxruntime': '0.1241', 'torchscript': '0.1516', 'xla': '0.1241', 'tvm': '0.1241'}

Usage:
    python reproduce/bug_0104.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0104.onnx")

def test_discrepancy_onnxruntime():
    """
    DISCREPANCY bug on onnxruntime: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.1241)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 32, 32]).astype(np.float32)}
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
        print(f"  [onnxruntime] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.1241)")
    else:
        print(f"  [onnxruntime] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.1241)")

def test_discrepancy_torchscript():
    """
    DISCREPANCY bug on torchscript: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.1516)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.torchscript_backend import TorchScriptBackend
        
        be = TorchScriptBackend()
    except Exception as e:
        print(f"  SKIP [torchscript] not installed: {e}")
        return
    res_opt   = be.run(model, inputs, optimized=True)
    res_noopt = be.run(model, inputs, optimized=False)
    if res_opt.crashed or res_noopt.crashed:
        print(f"  [torchscript] CRASH during run (unexpected for discrepancy bug)")
        return
    if res_opt.output is None or res_noopt.output is None:
        print(f"  [torchscript] NO OUTPUT")
        return
    a  = res_opt.output.flatten().astype(np.float32)
    b  = res_noopt.output.flatten().astype(np.float32)
    n  = min(a.size, b.size)
    rd = float(np.linalg.norm(a[:n] - b[:n])) / max(float(np.linalg.norm(b[:n])), 1e-8)
    if rd > 0.05:
        print(f"  [torchscript] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.1516)")
    else:
        print(f"  [torchscript] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.1516)")

def test_discrepancy_tvm():
    """
    DISCREPANCY bug on tvm: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.1241)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 32, 32]).astype(np.float32)}
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
        print(f"  [tvm] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.1241)")
    else:
        print(f"  [tvm] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.1241)")

def test_discrepancy_xla():
    """
    DISCREPANCY bug on xla: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 0.1241)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 32, 32]).astype(np.float32)}
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
        print(f"  [xla] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=0.1241)")
    else:
        print(f"  [xla] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=0.1241)")


if __name__ == "__main__":
    print("Reproducing bug_0104 [DISCREPANCY]")
    print("  patterns : branch/add_layernorm -> fusion/conv_bn_leakyrelu -> normalization/ada_layer_norm -> layout/double_transpose -> broadcast/where_mask -> fusion/grouped_conv_bn_relu")
    print("  input    : [1, 128, 32, 32]")
    test_discrepancy_onnxruntime()
    test_discrepancy_torchscript()
    test_discrepancy_tvm()
    test_discrepancy_xla()
