#!/usr/bin/env python3
"""
bug_0035 — CRASH bug
============================================================
Compiler class : onnxruntime
Affected       : onnxruntime
Total score    : 1.0045
Input shape    : [1, 64, 256]
Patterns       : branch/add_layernorm -> branch/add_layernorm -> branch/add_layernorm -> layout/squeeze_unsqueeze -> attention/rotary_embedding_attention -> normalization/rmsnorm

Bug type: CRASH
Crash errors:
  [onnxruntime+opt]: [ONNXRuntimeError] : 1 : FAIL : Node () Op (Slice) [ShapeInferenceError] Input axes has incorrect length
  [onnxruntime-opt]: [ONNXRuntimeError] : 1 : FAIL : Node () Op (Slice) [ShapeInferenceError] Input axes has incorrect length


Usage:
    python reproduce/bug_0035.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0035.onnx")

def test_crash_onnxruntime():
    """
    CRASH bug on onnxruntime.
    Expected error: '[ONNXRuntimeError] : 1 : FAIL : Node () Op (Slice) [ShapeInferenceError] Input axes has incorrect length'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 64, 256]).astype(np.float32)}
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


if __name__ == "__main__":
    print("Reproducing bug_0035 [CRASH]")
    print("  patterns : branch/add_layernorm -> branch/add_layernorm -> branch/add_layernorm -> layout/squeeze_unsqueeze -> attention/rotary_embedding_attention -> normalization/rmsnorm")
    print("  input    : [1, 64, 256]")
    test_crash_onnxruntime()
