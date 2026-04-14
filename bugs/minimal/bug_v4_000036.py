#!/usr/bin/env python3
"""
Bug ID     : bug_000036
Source     : Trion campaign v4 (minimized via delta-debug)
Compiler   : tvm, xla
Patterns   : MatMul, MatMul, MatMul, Resize
Root cause : tvm rel_L2=0.309
Minimal ops: MatMul -> MatMul -> MatMul -> Resize (4 ops, down from 5)
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
BACKENDS = ['tvm', 'xla']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_NAME = 'n300_resi_out'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 1, 32, 32]), 'n0_mm4d_w')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 1, 32, 32]), 'n100_mm4d_w')
    i_2 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 1, 32, 32]), 'n200_mm4d_w')
    i_3 = numpy_helper.from_array(np.array([], dtype=np.float32).reshape([0]), 'n300_resi_roi')
    i_4 = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32).reshape([4]), 'n300_resi_scales')
    return [i_0, i_1, i_2, i_3, i_4]


def _nodes():
    return [
        helper.make_node('MatMul', inputs=['model_input', 'n0_mm4d_w'], outputs=['n0_mm4d_out']),
        helper.make_node('MatMul', inputs=['n0_mm4d_out', 'n100_mm4d_w'], outputs=['n100_mm4d_out']),
        helper.make_node('MatMul', inputs=['n100_mm4d_out', 'n200_mm4d_w'], outputs=['n200_mm4d_out']),
        helper.make_node('Resize', inputs=['n200_mm4d_out', 'n300_resi_roi', 'n300_resi_scales'], outputs=['n300_resi_out'], coordinate_transformation_mode='half_pixel', mode='cubic'),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000036_min", inputs_info, outputs_info, initializer=_inits())
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
