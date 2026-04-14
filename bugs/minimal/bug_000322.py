#!/usr/bin/env python3
"""
Bug ID     : bug_000322
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : tensorflow
Patterns   : Pad
Root cause : tensorflow rel_L2=0.344
Minimal ops: Pad (1 ops, down from 13)
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
OUTPUT_NAME = 'n0_prc_pad'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64).reshape([8]), 'n0_prc_pads')
    return [i_0]


def _nodes():
    return [
        helper.make_node('Pad', inputs=['model_input', 'n0_prc_pads'], outputs=['n0_prc_pad'], mode='reflect'),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000322_min", inputs_info, outputs_info, initializer=_inits())
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
