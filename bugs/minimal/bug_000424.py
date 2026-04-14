#!/usr/bin/env python3
"""
Bug ID     : bug_000424
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : onnxruntime, openvino, tensorflow, xla
Patterns   : Flatten, Reshape, MatMul, BatchNormalization, Clip, CumSum
Root cause : onnxruntime rel_L2=1.030
Minimal ops: Flatten -> Reshape -> MatMul -> BatchNormalization -> Clip -> CumSum (6 ops, down from 8)
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
BACKENDS = ['onnxruntime', 'openvino', 'tensorflow', 'xla']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_NAME = 'n300_cs_out'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([1, 3, 32, 32], dtype=np.int64).reshape([4]), 'n0_far_b')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 1, 32, 32]), 'n100_mm4d_w')
    i_2 = numpy_helper.from_array(np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape([3]), 'n200_bnr6_sc')
    i_3 = numpy_helper.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape([3]), 'n200_bnr6_b')
    i_4 = numpy_helper.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape([3]), 'n200_bnr6_m')
    i_5 = numpy_helper.from_array(np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape([3]), 'n200_bnr6_v')
    i_6 = numpy_helper.from_array(np.array([0.0], dtype=np.float32).reshape([1]), 'n200_bnr6_zero')
    i_7 = numpy_helper.from_array(np.array([6.0], dtype=np.float32).reshape([1]), 'n200_bnr6_six')
    i_8 = numpy_helper.from_array(np.array([3], dtype=np.int64).reshape([]), 'n300_cs_ax')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8]


def _nodes():
    return [
        helper.make_node('Flatten', inputs=['model_input'], outputs=['n0_far_fl'], axis=2),
        helper.make_node('Reshape', inputs=['n0_far_fl', 'n0_far_b'], outputs=['n0_far_out']),
        helper.make_node('MatMul', inputs=['n0_far_out', 'n100_mm4d_w'], outputs=['n100_mm4d_out']),
        helper.make_node('BatchNormalization', inputs=['n100_mm4d_out', 'n200_bnr6_sc', 'n200_bnr6_b', 'n200_bnr6_m', 'n200_bnr6_v'], outputs=['n200_bnr6_bn'], epsilon=9.999999747378752e-06),
        helper.make_node('Clip', inputs=['n200_bnr6_bn', 'n200_bnr6_zero', 'n200_bnr6_six'], outputs=['n200_bnr6_out']),
        helper.make_node('CumSum', inputs=['n200_bnr6_out', 'n300_cs_ax'], outputs=['n300_cs_out']),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000424_min", inputs_info, outputs_info, initializer=_inits())
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
