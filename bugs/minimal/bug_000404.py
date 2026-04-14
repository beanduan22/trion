#!/usr/bin/env python3
"""
Bug ID     : bug_000404
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : tensorflow, xla
Patterns   : Resize
Root cause : tensorflow rel_L2=0.339
Minimal ops: Resize (1 ops, down from 8)
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
BACKENDS = ['tensorflow', 'xla']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_NAME = 'n0_resi_out'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([], dtype=np.float32).reshape([0]), 'n0_resi_roi')
    i_1 = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32).reshape([4]), 'n0_resi_scales')
    return [i_0, i_1]


def _nodes():
    return [
        helper.make_node('Resize', inputs=['model_input', 'n0_resi_roi', 'n0_resi_scales'], outputs=['n0_resi_out'], coordinate_transformation_mode='align_corners', mode='linear'),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000404_min", inputs_info, outputs_info, initializer=_inits())
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
