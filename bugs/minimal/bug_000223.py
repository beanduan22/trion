#!/usr/bin/env python3
"""
Bug ID     : bug_000223
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : tensorflow
Patterns   : Expand, Add, Pad
Root cause : tensorflow rel_L2=0.337
Minimal ops: Expand -> Add -> Pad (3 ops, down from 9)
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
OUTPUT_NAME = 'n100_prc_pad'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape([1, 3, 1, 1]), 'n0_era_src')
    i_1 = numpy_helper.from_array(np.array([1, 3, 32, 32], dtype=np.int64).reshape([4]), 'n0_era_tgt')
    i_2 = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64).reshape([8]), 'n100_prc_pads')
    return [i_0, i_1, i_2]


def _nodes():
    return [
        helper.make_node('Expand', inputs=['n0_era_src', 'n0_era_tgt'], outputs=['n0_era_ex']),
        helper.make_node('Add', inputs=['model_input', 'n0_era_ex'], outputs=['n0_era_out']),
        helper.make_node('Pad', inputs=['n0_era_out', 'n100_prc_pads'], outputs=['n100_prc_pad'], mode='reflect'),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000223_min", inputs_info, outputs_info, initializer=_inits())
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
