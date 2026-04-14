#!/usr/bin/env python3
"""
Bug ID     : bug_000290
Source     : Trion campaign v4 (minimized via delta-debug)
Compiler   : onnxruntime, openvino, tvm, xla
Patterns   : Abs, Mul, CumSum
Root cause : onnxruntime rel_L2=1.374
Minimal ops: Abs -> Mul -> CumSum (3 ops, down from 13)
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
BACKENDS = ['onnxruntime', 'openvino', 'tvm', 'xla']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 16, 128]
OUTPUT_NAME = 'n100_cs_out'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([2], dtype=np.int64).reshape([]), 'n100_cs_ax')
    return [i_0]


def _nodes():
    return [
        helper.make_node('Abs', inputs=['model_input'], outputs=['n0_abs__a']),
        helper.make_node('Mul', inputs=['n0_abs__a', 'model_input'], outputs=['n0_abs__out']),
        helper.make_node('CumSum', inputs=['n0_abs__out', 'n100_cs_ax'], outputs=['n100_cs_out']),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000290_min", inputs_info, outputs_info, initializer=_inits())
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = IR_VERSION
    return m.SerializeToString()


def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 16, 128])


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
