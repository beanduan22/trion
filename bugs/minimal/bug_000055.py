#!/usr/bin/env python3
"""
Bug ID     : bug_000055
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : tensorflow
Patterns   : MatMul, Conv, Tanh, HardSwish, Mul, Abs, Add, Pow...
Root cause : tensorflow rel_L2=0.336
Minimal ops: MatMul -> Conv -> Tanh -> HardSwish -> Mul -> Abs -> Add -> Pow -> Sqrt -> Pad (10 ops, down from 11)
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
OUTPUT_NAME = 'n400_rfpcv_pad'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 1, 32, 32]), 'n0_mm4d_w')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "/rUCPw6IhD1oXMs++KIyPs76Xz9Gxgy/8d/IPk/sCL7fawg/4jIZvjHdxj9vc5w/BO39Pmyv2b6B"
        "a2A9AN4bvmEx2D6vl6e/hlk2v5WFST/sjkQ/Xvg3vpFSbL4D2aA/gM5JvverHz8HjU8/UF6jv4kM"
        "BD7xytY/B4kvP8+6bL3LmwU+DZgwPmMhUj+j5CG/phWXv/yTHUBZNbK+vm1XvnorXL80WfU+6rsl"
        "v3XdTD/vJny/FOtNP9M0XL/4vqa+PIAYP+e7Rr9tRHA/g9Wbv+n5gT+RCtq+YJUWv9aKFz+yXGC/"
        "YcVAvSwt8T90gIE/BLGnv3+mfb/Vp0E/FY5SvvdvGMAYlVi+/MqoPplfBj8Hdh9AArPvv0wftD5i"
        "x0A+IqdivjOFbD7wonk9m4WFvqtXM79+s6a+IWjyPqmeFT8hOVE9O6wIP/Cx7T6BgAy/EaK4PpUN"
        "t7716XS9TgoYP36ZVD5kEQk/hClWv63cgj7/8sg8DHPRPr84pr7GNxY+"
    ), dtype=np.float32).copy().reshape([32, 3, 1, 1]), 'n100_cth_w')
    i_2 = numpy_helper.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape([32]), 'n100_cth_b')
    i_3 = numpy_helper.from_array(np.array([1.0], dtype=np.float32).reshape([1]), 'n200_hard_one')
    i_4 = numpy_helper.from_array(np.array([2.0], dtype=np.float32).reshape([1]), 'n300_pwc_two')
    i_5 = numpy_helper.from_array(np.array([9.999999974752427e-07], dtype=np.float32).reshape([1]), 'n300_pwc_eps')
    i_6 = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64).reshape([8]), 'n400_rfpcv_pads')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6]


def _nodes():
    return [
        helper.make_node('MatMul', inputs=['model_input', 'n0_mm4d_w'], outputs=['n0_mm4d_out']),
        helper.make_node('Conv', inputs=['n0_mm4d_out', 'n100_cth_w', 'n100_cth_b'], outputs=['n100_cth_cv'], kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
        helper.make_node('Tanh', inputs=['n100_cth_cv'], outputs=['n100_cth_out']),
        helper.make_node('HardSwish', inputs=['n100_cth_out'], outputs=['n200_hard_act']),
        helper.make_node('Mul', inputs=['n200_hard_act', 'n200_hard_one'], outputs=['n200_hard_out']),
        helper.make_node('Abs', inputs=['n200_hard_out'], outputs=['n300_pwc_abs']),
        helper.make_node('Add', inputs=['n300_pwc_abs', 'n300_pwc_eps'], outputs=['n300_pwc_add']),
        helper.make_node('Pow', inputs=['n300_pwc_add', 'n300_pwc_two'], outputs=['n300_pwc_sq']),
        helper.make_node('Sqrt', inputs=['n300_pwc_sq'], outputs=['n300_pwc_out']),
        helper.make_node('Pad', inputs=['n300_pwc_out', 'n400_rfpcv_pads'], outputs=['n400_rfpcv_pad'], mode='reflect'),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000055_min", inputs_info, outputs_info, initializer=_inits())
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
