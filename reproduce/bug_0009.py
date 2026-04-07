#!/usr/bin/env python3
"""
bug_0009 — CRASH bug
============================================================
Compiler class : tflite
Affected       : tflite, tvm, xla
Total score    : 5.0000
Input shape    : [1, 3, 32, 32]
Patterns       : fusion/conv_transpose_bn_relu -> broadcast/swish -> broadcast/abs_neg_relu -> fusion/conv_relu_conv_bn -> broadcast/clipped_affine -> normalization/variance_whitening

Bug type: CRASH
Crash errors:
  [xla+opt]: conv_general_dilated lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size, but 3 // 1 != 256.
  [xla-opt]: conv_general_dilated lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size, but 3 // 1 != 256.
  [tvm+opt]: conv_general_dilated lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size, but 3 // 1 != 256.
  [tvm-opt]: conv_general_dilated lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size, but 3 // 1 != 256.
  [tflite+opt]: Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:319:1: error: 'tf.Conv2DBackpropInput' op is neither a custom op nor a flex op
    if op == "ConvTranspose"
  [tflite-opt]: Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:319:1: error: 'tf.Conv2DBackpropInput' op is neither a custom op nor a flex op
    if op == "ConvTranspose"


Usage:
    python reproduce/bug_0009.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0009.onnx")

def test_crash_tflite():
    """
    CRASH bug on tflite.
    Expected error: 'Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:319:1: error: \'tf.Conv2DBackpropInput\' op is neither a custom op nor a flex op\n    if op == "ConvTranspose"'
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

def test_crash_tvm():
    """
    CRASH bug on tvm.
    Expected error: 'conv_general_dilated lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size, but 3 // 1 != 256.'
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
    Expected error: 'conv_general_dilated lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size, but 3 // 1 != 256.'
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


if __name__ == "__main__":
    print("Reproducing bug_0009 [CRASH]")
    print("  patterns : fusion/conv_transpose_bn_relu -> broadcast/swish -> broadcast/abs_neg_relu -> fusion/conv_relu_conv_bn -> broadcast/clipped_affine -> normalization/variance_whitening")
    print("  input    : [1, 3, 32, 32]")
    test_crash_tflite()
    test_crash_tvm()
    test_crash_xla()
