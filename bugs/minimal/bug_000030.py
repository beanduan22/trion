#!/usr/bin/env python3
"""
Bug ID     : bug_000030
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : tensorflow
Patterns   : Pad
Root cause : tensorflow rel_L2=0.329
Minimal ops: Pad (1 ops, down from 8)
Tolerance  : 0.1

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
from __future__ import annotations
import base64, io, os, sys
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
except ImportError as e:
    print(f"missing deps: {e}"); sys.exit(2)

TOLERANCE = 0.1
BACKENDS = ['tensorflow']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_NAME = 'n0_prma_out'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([0, 0, 2, 0, 0, 0, 0, 2], dtype=np.int64).reshape([8]), 'n0_prma_pads')
    return [i_0]


def _nodes():
    return [
        helper.make_node('Pad', inputs=['model_input', 'n0_prma_pads'], outputs=['n0_prma_out'], mode='reflect'),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000030_min", inputs_info, outputs_info, initializer=_inits())
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = IR_VERSION
    return m.SerializeToString()


def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 3, 32, 32])


def _ref_pytorch(mb, x):
    import torch, onnx2torch
    m = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    with torch.no_grad():
        out = m(torch.from_numpy(x))
    if isinstance(out, (list, tuple)): out = out[0]
    return out.detach().cpu().float().numpy().ravel()


def _rel_l2(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-12))


def _run(backend, mb, x):
    try:
        if backend == "onnxruntime":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(mb, opts, providers=["CPUExecutionProvider"])
            return np.asarray(sess.run(None, {INPUT_NAME: x})[0]).ravel(), None
        if backend == "openvino":
            import openvino as ov
            core = ov.Core()
            compiled = core.compile_model(core.read_model(io.BytesIO(mb)), "CPU")
            out = compiled([x])
            return np.asarray(list(out.values())[0]).ravel(), None
        if backend in ("tensorflow", "xla"):
            sys.path.insert(0, "/home/binduan/myspace/trion")
            from trion.oracle.tf_backend import TFBackend
            r = TFBackend().run(onnx.load_from_string(mb), {INPUT_NAME: x}, optimized=(backend == "xla"))
            return (r.output.ravel(), None) if r.output is not None else (None, r.error or "fail")
        if backend == "tvm":
            sys.path.insert(0, "/home/binduan/myspace/trion")
            from trion.oracle.tvm_backend import TVMBackend
            r = TVMBackend().run(onnx.load_from_string(mb), {INPUT_NAME: x}, optimized=True)
            return (r.output.ravel(), None) if r.output is not None else (None, r.error or "fail")
        if backend == "torchscript":
            import torch, onnx2torch
            m = onnx2torch.convert(onnx.load_from_string(mb)).eval()
            t = torch.from_numpy(x)
            scripted = torch.jit.trace(m, (t,))
            frozen = torch.jit.optimize_for_inference(torch.jit.freeze(scripted))
            with torch.no_grad():
                out = frozen(t)
            if isinstance(out, (list, tuple)): out = out[0]
            return out.detach().numpy().ravel(), None
        if backend == "torch_compile":
            import torch, onnx2torch
            m = onnx2torch.convert(onnx.load_from_string(mb)).eval()
            compiled = torch.compile(m, mode="reduce-overhead", fullgraph=False)
            with torch.no_grad():
                out = compiled(torch.from_numpy(x))
            if isinstance(out, (list, tuple)): out = out[0]
            return out.detach().numpy().ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:100]}"
    return None, "no driver"


def main():
    mb = build_model()
    x = _input()
    try:
        ref = _ref_pytorch(mb, x)
    except Exception as e:
        print(f"ref failed: {e}"); return 2
    any_bug = False
    print(f"Bug ID: minimized")
    print(f"Backends: {BACKENDS}  Tolerance: {TOLERANCE}")
    for b in BACKENDS:
        out, err = _run(b, mb, x)
        if err:
            print(f"  [{b}] CRASH: {err}   -> BUG REPRODUCED")
            any_bug = True; continue
        d = _rel_l2(out, ref)
        verdict = "REPRODUCED" if d > TOLERANCE else "ok"
        print(f"  [{b}] rel_L2 vs pytorch_ref = {d:.4e}   -> {verdict}")
        if d > TOLERANCE: any_bug = True
    PASS = not any_bug
    print(f"PASS={PASS}")
    if not PASS:
        print("BUG REPRODUCED"); return 0
    print("not reproduced"); return 1


if __name__ == "__main__":
    sys.exit(main())
