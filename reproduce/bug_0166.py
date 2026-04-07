#!/usr/bin/env python3
"""
bug_0166 — CRASH bug
============================================================
Compiler class : tflite
Affected       : tflite
Total score    : 1.0000
Input shape    : [1, 3, 32, 32]
Patterns       : branch/spatial_attention_cbam -> layout/dilated_max_pool -> layout/reshape_batched_matmul -> broadcast/abs_neg_relu -> branch/glu -> fusion/matmul_bias_sigmoid

Bug type: CRASH
Crash errors:
  [tflite+opt]: Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:358:1: error: 'tf.ExtractImagePatches' op is neither a custom op nor a flex op
        return _pool(node, g
  [tflite-opt]: Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:358:1: error: 'tf.ExtractImagePatches' op is neither a custom op nor a flex op
        return _pool(node, g


Usage:
    python reproduce/bug_0166.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0166.onnx")

def test_crash_tflite():
    """
    CRASH bug on tflite.
    Expected error: "Could not translate MLIR to FlatBuffer./home/binduan/myspace/trion/trion/oracle/_onnx_ops.py:358:1: error: 'tf.ExtractImagePatches' op is neither a custom op nor a flex op\n        return _pool(node, g"
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


if __name__ == "__main__":
    print("Reproducing bug_0166 [CRASH]")
    print("  patterns : branch/spatial_attention_cbam -> layout/dilated_max_pool -> layout/reshape_batched_matmul -> broadcast/abs_neg_relu -> branch/glu -> fusion/matmul_bias_sigmoid")
    print("  input    : [1, 3, 32, 32]")
    test_crash_tflite()
