#!/usr/bin/env python3
"""
Bug ID     : bug_000078
Source     : Trion campaign v4 (minimized via delta-debug)
Compiler   : onnxruntime, openvino
Patterns   : Mul, ReduceMean, Add, Sqrt, Div, Mul, Mul, Resize
Root cause : onnxruntime rel_L2=1.224
Minimal ops: Mul -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Mul -> Resize (8 ops, down from 14)
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
BACKENDS = ['onnxruntime', 'openvino']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_NAME = 'n100_resi_out'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape([32]), 'n0_crms_g')
    i_1 = numpy_helper.from_array(np.array([1.012573003768921, 0.9867895245552063, 1.0640422105789185, 1.0104900598526, 0.9464330673217773, 1.0361595153808594, 1.1304000616073608, 1.0947080850601196, 0.92962646484375, 0.8734578490257263, 0.9376725554466248, 1.004132628440857, 0.7674969434738159, 0.9781208038330078, 0.8754088878631592, 0.9267732501029968, 0.9455741047859192, 0.9683699607849121, 1.0411630868911743, 1.1042513847351074, 0.9871465563774109, 1.1366463899612427, 0.9334805607795715, 1.0351510047912598, 1.0903470516204834, 1.0094012022018433, 0.9256500601768494, 0.9078274369239807, 0.9542273879051208, 1.0220195055007935, 0.8990381956100464, 0.979082465171814], dtype=np.float32).reshape([32]), 'n0_crms_g2')
    i_2 = numpy_helper.from_array(np.array([9.999999974752427e-07], dtype=np.float32).reshape([1]), 'n0_crms_eps')
    i_3 = numpy_helper.from_array(np.array([], dtype=np.float32).reshape([0]), 'n100_resi_roi')
    i_4 = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32).reshape([4]), 'n100_resi_scales')
    return [i_0, i_1, i_2, i_3, i_4]


def _nodes():
    return [
        helper.make_node('Mul', inputs=['model_input', 'model_input'], outputs=['n0_crms_sq']),
        helper.make_node('ReduceMean', inputs=['n0_crms_sq'], outputs=['n0_crms_ms'], axes=[-1], keepdims=1),
        helper.make_node('Add', inputs=['n0_crms_ms', 'n0_crms_eps'], outputs=['n0_crms_add']),
        helper.make_node('Sqrt', inputs=['n0_crms_add'], outputs=['n0_crms_rt']),
        helper.make_node('Div', inputs=['model_input', 'n0_crms_rt'], outputs=['n0_crms_n']),
        helper.make_node('Mul', inputs=['n0_crms_n', 'n0_crms_g'], outputs=['n0_crms_sc']),
        helper.make_node('Mul', inputs=['n0_crms_sc', 'n0_crms_g2'], outputs=['n0_crms_out']),
        helper.make_node('Resize', inputs=['n0_crms_out', 'n100_resi_roi', 'n100_resi_scales'], outputs=['n100_resi_out'], coordinate_transformation_mode='half_pixel', mode='nearest', nearest_mode='ceil'),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000078_min", inputs_info, outputs_info, initializer=_inits())
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
