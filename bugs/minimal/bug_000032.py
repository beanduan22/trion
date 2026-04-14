#!/usr/bin/env python3
"""
Bug ID     : bug_000032
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : tensorflow
Patterns   : Mul, Add, Relu, Pow, Mul, Sub, Add, Reshape...
Root cause : tensorflow crash
Minimal ops: Mul -> Add -> Relu -> Pow -> Mul -> Sub -> Add -> Reshape -> Transpose -> Reshape -> Conv (11 ops, down from 12)
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
OUTPUT_NAME = 'n300_s2d_out'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape([1, 3, 1, 1]), 'n0_mar_scale')
    i_1 = numpy_helper.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape([1, 3, 1, 1]), 'n0_mar_bias')
    i_2 = numpy_helper.from_array(np.array([2], dtype=np.int64).reshape([1]), 'n100_pi2_two')
    i_3 = numpy_helper.from_array(np.array([1, 3, 16, 2, 16, 2], dtype=np.int64).reshape([6]), 'n300_s2d_s1')
    i_4 = numpy_helper.from_array(np.array([1, 12, 16, 16], dtype=np.int64).reshape([4]), 'n300_s2d_s2')
    i_5 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([64, 12, 1, 1]), 'n300_s2d_w')
    i_6 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    ), dtype=np.float32).copy().reshape([64]), 'n300_s2d_b')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6]


def _nodes():
    return [
        helper.make_node('Mul', inputs=['model_input', 'n0_mar_scale'], outputs=['n0_mar_mul']),
        helper.make_node('Add', inputs=['n0_mar_mul', 'n0_mar_bias'], outputs=['n0_mar_add']),
        helper.make_node('Relu', inputs=['n0_mar_add'], outputs=['n0_mar_out']),
        helper.make_node('Pow', inputs=['n0_mar_out', 'n100_pi2_two'], outputs=['n100_pi2_pw']),
        helper.make_node('Mul', inputs=['n100_pi2_pw', 'n0_mar_out'], outputs=['n100_pi2_out']),
        helper.make_node('Sub', inputs=['n100_pi2_out', 'n100_pi2_out'], outputs=['n200_ssz_z']),
        helper.make_node('Add', inputs=['n100_pi2_out', 'n200_ssz_z'], outputs=['n200_ssz_out']),
        helper.make_node('Reshape', inputs=['n200_ssz_out', 'n300_s2d_s1'], outputs=['n300_s2d_r1']),
        helper.make_node('Transpose', inputs=['n300_s2d_r1'], outputs=['n300_s2d_tr'], perm=[0, 1, 3, 2, 5, 4]),
        helper.make_node('Reshape', inputs=['n300_s2d_tr', 'n300_s2d_s2'], outputs=['n300_s2d_r2']),
        helper.make_node('Conv', inputs=['n300_s2d_r2', 'n300_s2d_w', 'n300_s2d_b'], outputs=['n300_s2d_out'], kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000032_min", inputs_info, outputs_info, initializer=_inits())
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
