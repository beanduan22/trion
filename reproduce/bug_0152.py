#!/usr/bin/env python3
"""
bug_0152 — CRASH bug
============================================================
Compiler class : xla
Affected       : torch_compile, torchscript, tvm, xla
Total score    : 4.0002
Input shape    : [1, 128, 16, 16]
Patterns       : broadcast/sub_mul_add -> branch/channel_gating_branch -> normalization/rmsnorm -> layout/reshape_transpose_reshape -> constant/pow_canonical -> layout/depth_to_space_block

Bug type: CRASH
Crash errors:
  [torchscript+opt]: DepthToSpace for mode other than CRD is not implemented
  [torchscript-opt]: DepthToSpace for mode other than CRD is not implemented
  [torch_compile+opt]: DepthToSpace for mode other than CRD is not implemented
  [torch_compile-opt]: DepthToSpace for mode other than CRD is not implemented
  [xla+opt]: DepthToSpace for mode other than CRD is not implemented
  [xla-opt]: DepthToSpace for mode other than CRD is not implemented
  [tvm+opt]: DepthToSpace for mode other than CRD is not implemented
  [tvm-opt]: DepthToSpace for mode other than CRD is not implemented


Usage:
    python reproduce/bug_0152.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "bugs", "bug_0152.onnx")

def test_crash_torch_compile():
    """
    CRASH bug on torch_compile.
    Expected error: 'DepthToSpace for mode other than CRD is not implemented'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 16, 16]).astype(np.float32)}
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
    Expected error: 'DepthToSpace for mode other than CRD is not implemented'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 16, 16]).astype(np.float32)}
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

def test_crash_tvm():
    """
    CRASH bug on tvm.
    Expected error: 'DepthToSpace for mode other than CRD is not implemented'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 16, 16]).astype(np.float32)}
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
    Expected error: 'DepthToSpace for mode other than CRD is not implemented'
    """
    import onnx, numpy as np
    model  = onnx.load(ONNX_PATH)
    inputs = {model.graph.input[0].name: np.random.default_rng(42).standard_normal([1, 128, 16, 16]).astype(np.float32)}
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
    print("Reproducing bug_0152 [CRASH]")
    print("  patterns : broadcast/sub_mul_add -> branch/channel_gating_branch -> normalization/rmsnorm -> layout/reshape_transpose_reshape -> constant/pow_canonical -> layout/depth_to_space_block")
    print("  input    : [1, 128, 16, 16]")
    test_crash_torch_compile()
    test_crash_torchscript()
    test_crash_tvm()
    test_crash_xla()
