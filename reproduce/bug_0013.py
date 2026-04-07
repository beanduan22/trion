#!/usr/bin/env python3
"""
bug_0013 — CRASH bug
============================================================
Compiler class : tflite
Affected       : onnxruntime, openvino, tensorflow, tflite, torch_compile, torchscript
Total score    : 2.0007
Input shape    : [1, 3, 32, 32]
Patterns       : constant/learned_temperature_scale -> constant/cast_roundtrip -> layout/ceil_mode_avg_pool_conv -> fusion/conv_asym_pad_bn -> fusion/conv_bn_relu -> normalization/rmsnorm

Bug type: CRASH
Crash errors:
  [pytorch_eager]: [ONNXRuntimeError] : 1 : FAIL : Node () Op (Mul) [ShapeInferenceError] Incompatible dimensions
  [onnxruntime+opt]: [ONNXRuntimeError] : 1 : FAIL : Node () Op (Mul) [ShapeInferenceError] Incompatible dimensions
  [onnxruntime-opt]: [ONNXRuntimeError] : 1 : FAIL : Node () Op (Mul) [ShapeInferenceError] Incompatible dimensions
  [torchscript+opt]: The size of tensor a (16) must match the size of tensor b (15) at non-singleton dimension 3
  [torchscript-opt]: The size of tensor a (16) must match the size of tensor b (15) at non-singleton dimension 3
  [torch_compile-opt]: The size of tensor a (16) must match the size of tensor b (15) at non-singleton dimension 3
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
  [openvino+opt]: Check 'false' failed at src/frontends/onnx/frontend/src/frontend.cpp:387:
FrontEnd API failed with GeneralFailure:
Errors during ONNX translation:
[ONNX Frontend] Conversion failed for Mul-17
While va
  [openvino-opt]: Check 'false' failed at src/frontends/onnx/frontend/src/frontend.cpp:387:
FrontEnd API failed with GeneralFailure:
Errors during ONNX translation:
[ONNX Frontend] Conversion failed for Mul-17
While va


Usage:
    python reproduce/bug_0013.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0013.onnx")

def test_crash_onnxruntime():
    """
    CRASH bug on onnxruntime.
    Expected error: '[ONNXRuntimeError] : 1 : FAIL : Node () Op (Mul) [ShapeInferenceError] Incompatible dimensions'
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
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [onnxruntime+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [onnxruntime+{tag}] no crash (may be fixed)")

def test_crash_openvino():
    """
    CRASH bug on openvino.
    Expected error: "Check 'false' failed at src/frontends/onnx/frontend/src/frontend.cpp:387:\nFrontEnd API failed with GeneralFailure:\nErrors during ONNX translation:\n[ONNX Frontend] Conversion failed for Mul-17\nWhile va"
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
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [openvino+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [openvino+{tag}] no crash (may be fixed)")

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
    Expected error: 'The size of tensor a (16) must match the size of tensor b (15) at non-singleton dimension 3'
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

def test_crash_torchscript():
    """
    CRASH bug on torchscript.
    Expected error: 'The size of tensor a (16) must match the size of tensor b (15) at non-singleton dimension 3'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 3, 32, 32]).astype(np.float32)}
    try:
        from trion.oracle.torchscript_backend import TorchScriptBackend
        
        be = TorchScriptBackend()
    except Exception as e:
        print(f"  SKIP [torchscript] not installed: {e}")
        return
    for opt in [True, False]:
        tag = "opt" if opt else "noopt"
        res = be.run(model, inputs, optimized=opt)
        if res.crashed or res.output is None:
            print(f"  [torchscript+{tag}] CRASH confirmed: {(res.error or '')[:200]}")
        else:
            print(f"  [torchscript+{tag}] no crash (may be fixed)")


if __name__ == "__main__":
    print("Reproducing bug_0013 [CRASH]")
    print("  patterns : constant/learned_temperature_scale -> constant/cast_roundtrip -> layout/ceil_mode_avg_pool_conv -> fusion/conv_asym_pad_bn -> fusion/conv_bn_relu -> normalization/rmsnorm")
    print("  input    : [1, 3, 32, 32]")
    test_crash_onnxruntime()
    test_crash_openvino()
    test_crash_tensorflow()
    test_crash_tflite()
    test_crash_torch_compile()
    test_crash_torchscript()
