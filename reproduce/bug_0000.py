#!/usr/bin/env python3
"""
bug_0000 — CRASH+DISCREPANCY bug
============================================================
Compiler class : tflite
Affected       : tensorflow, tflite, torch_compile, tvm, xla
Total score    : 6.0000
Input shape    : [1, 3, 32, 32]
Patterns       : layout/dilated_max_pool -> branch/concat_conv -> broadcast/hard_clamp_norm -> fusion/depthwise_conv_bn_relu -> layout/reshape_batched_matmul -> layout/gather_reshape

Bug type: CRASH+DISCREPANCY
Crash errors:
  [xla+opt]: Cannot concatenate arrays with shapes that differ in dimensions other than the one being concatenated: concatenating along dimension 1 for shapes (1, 3, 32, 32), (1, 64, 30, 30).
  [xla-opt]: Cannot concatenate arrays with shapes that differ in dimensions other than the one being concatenated: concatenating along dimension 1 for shapes (1, 3, 32, 32), (1, 64, 30, 30).
  [tvm+opt]: Cannot concatenate arrays with shapes that differ in dimensions other than the one being concatenated: concatenating along dimension 1 for shapes (1, 3, 32, 32), (1, 64, 30, 30).
  [tvm-opt]: Cannot concatenate arrays with shapes that differ in dimensions other than the one being concatenated: concatenating along dimension 1 for shapes (1, 3, 32, 32), (1, 64, 30, 30).
  [tensorflow+opt]: in user code:

    File "/home/binduan/myspace/trion/trion/oracle/tf_backend.py", line 61, in model_fn  *
        results = dispatch_op(node, values, tnp)
    File "/home/binduan/myspace/trion/trion/o
  [tensorflow-opt]: in user code:

    File "/home/binduan/myspace/trion/trion/oracle/tf_backend.py", line 61, in model_fn  *
        results = dispatch_op(node, values, tnp)
    File "/home/binduan/myspace/trion/trion/o
  [tflite+opt]: in user code:

    File "/home/binduan/myspace/trion/trion/oracle/tflite_backend.py", line 60, in model_fn  *
        results = dispatch_op(node, values, tnp)
    File "/home/binduan/myspace/trion/tri
  [tflite-opt]: in user code:

    File "/home/binduan/myspace/trion/trion/oracle/tflite_backend.py", line 60, in model_fn  *
        results = dispatch_op(node, values, tnp)
    File "/home/binduan/myspace/trion/tri
Discrepancy (rel_diff opt vs noopt): {'torch_compile': '1.0000'}

Usage:
    python reproduce/bug_0000.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0000.onnx")

def test_crash_tensorflow():
    """
    CRASH bug on tensorflow.
    Expected error: 'in user code:\n\n    File "/home/binduan/myspace/trion/trion/oracle/tf_backend.py", line 61, in model_fn  *\n        results = dispatch_op(node, values, tnp)\n    File "/home/binduan/myspace/trion/trion/o'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.tf_backend import TFBackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = TFBackend(cfg)
    except Exception as e:
        print(f"  SKIP [tensorflow] not installed: {e}")
        return
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [tensorflow+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [tensorflow+{tag}] no crash (may be fixed)")

def test_crash_tflite():
    """
    CRASH bug on tflite.
    Expected error: 'in user code:\n\n    File "/home/binduan/myspace/trion/trion/oracle/tflite_backend.py", line 60, in model_fn  *\n        results = dispatch_op(node, values, tnp)\n    File "/home/binduan/myspace/trion/tri'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.tflite_backend import TFLiteBackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = TFLiteBackend(cfg)
    except Exception as e:
        print(f"  SKIP [tflite] not installed: {e}")
        return
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [tflite+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [tflite+{tag}] no crash (may be fixed)")

def test_crash_torch_compile():
    """
    CRASH bug on torch_compile.
    Expected error: ''
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.torch_compile_backend import TorchCompileBackend
        
        be = TorchCompileBackend()
    except Exception as e:
        print(f"  SKIP [torch_compile] not installed: {e}")
        return
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [torch_compile+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [torch_compile+{tag}] no crash (may be fixed)")

def test_crash_tvm():
    """
    CRASH bug on tvm.
    Expected error: 'Cannot concatenate arrays with shapes that differ in dimensions other than the one being concatenated: concatenating along dimension 1 for shapes (1, 3, 32, 32), (1, 64, 30, 30).'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.tvm_backend import TVMBackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = TVMBackend(cfg)
    except Exception as e:
        print(f"  SKIP [tvm] not installed: {e}")
        return
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [tvm+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [tvm+{tag}] no crash (may be fixed)")

def test_crash_xla():
    """
    CRASH bug on xla.
    Expected error: 'Cannot concatenate arrays with shapes that differ in dimensions other than the one being concatenated: concatenating along dimension 1 for shapes (1, 3, 32, 32), (1, 64, 30, 30).'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.xla_backend import XLABackend
            from trion.config import TrionConfig; cfg = TrionConfig()
        be = XLABackend(cfg)
    except Exception as e:
        print(f"  SKIP [xla] not installed: {e}")
        return
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [xla+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [xla+{tag}] no crash (may be fixed)")

def test_discrepancy_torch_compile():
    """
    DISCREPANCY bug on torch_compile: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
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
        print(f"  [torch_compile] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=1.0000)")
    else:
        print(f"  [torch_compile] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=1.0000)")


if __name__ == "__main__":
    print("Reproducing bug_0000 [CRASH+DISCREPANCY]")
    print("  patterns : layout/dilated_max_pool -> branch/concat_conv -> broadcast/hard_clamp_norm -> fusion/depthwise_conv_bn_relu -> layout/reshape_batched_matmul -> layout/gather_reshape")
    print("  input    : [1, 3, 32, 32]")
    test_crash_tensorflow()
    test_crash_tflite()
    test_crash_torch_compile()
    test_crash_tvm()
    test_crash_xla()
    test_discrepancy_torch_compile()
