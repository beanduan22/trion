#!/usr/bin/env python3
"""
bug_0017 — CRASH+DISCREPANCY bug
============================================================
Compiler class : tflite
Affected       : onnxruntime, tensorflow, tflite, torch_compile, tvm, xla
Total score    : 12.0000
Input shape    : [1, 3, 32, 32]
Patterns       : branch/residual_add_relu -> constant/div_by_constant -> normalization/rmsnorm -> normalization/variance_whitening -> normalization/spatial_reduce_mean -> layout/dilated_max_pool

Bug type: CRASH+DISCREPANCY
Crash errors:
  [tensorflow+opt]: XLA does not support pooling ops with explicit padding.
	 [[{{node MaxPool2d}}]]
	tf2xla conversion failed while converting __inference_model_fn_154445[_XlaMustCompile=true,config_proto=13561319589895
  [tflite+opt]: Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:358:1: error: 'tf.MaxPool' op is neither a custom op nor a flex op
        return _pool(node, get, F)
^
/ho
  [tflite-opt]: Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:358:1: error: 'tf.MaxPool' op is neither a custom op nor a flex op
        return _pool(node, get, F)
^
/ho
Discrepancy (rel_diff opt vs noopt): {'onnxruntime': '1.0000', 'torch_compile': '1.0000', 'xla': '1.0000', 'tvm': '1.0000', 'tensorflow': '1.0000'}

Usage:
    python reproduce/bug_0017.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0017.onnx")

def test_crash_tensorflow():
    """
    CRASH bug on tensorflow.
    Expected error: 'XLA does not support pooling ops with explicit padding.\n\t [[{{node MaxPool2d}}]]\n\ttf2xla conversion failed while converting __inference_model_fn_154445[_XlaMustCompile=true,config_proto=13561319589895'
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
    Expected error: "Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:358:1: error: 'tf.MaxPool' op is neither a custom op nor a flex op\n        return _pool(node, get, F)\n^\n/ho"
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

def test_discrepancy_onnxruntime():
    """
    DISCREPANCY bug on onnxruntime: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
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

def test_discrepancy_tensorflow():
    """
    DISCREPANCY bug on tensorflow: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
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
        print(f"  [tensorflow] DISCREPANCY confirmed  rel_diff={rd:.4f}  (expected>0.05, stored=1.0000)")
    else:
        print(f"  [tensorflow] no discrepancy  rel_diff={rd:.4f}  (may be fixed, stored=1.0000)")

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

def test_discrepancy_tvm():
    """
    DISCREPANCY bug on tvm: optimized output differs from non-optimized.
    Expected rel_diff(opt_output, noopt_output) > 0.05  (stored: 1.0000)
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
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
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
    print("Reproducing bug_0017 [CRASH+DISCREPANCY]")
    print("  patterns : branch/residual_add_relu -> constant/div_by_constant -> normalization/rmsnorm -> normalization/variance_whitening -> normalization/spatial_reduce_mean -> layout/dilated_max_pool")
    print("  input    : [1, 3, 32, 32]")
    test_crash_tensorflow()
    test_crash_tflite()
    test_discrepancy_onnxruntime()
    test_discrepancy_tensorflow()
    test_discrepancy_torch_compile()
    test_discrepancy_tvm()
    test_discrepancy_xla()
