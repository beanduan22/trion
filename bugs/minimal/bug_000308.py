#!/usr/bin/env python3
"""
Bug ID     : bug_000308
Source     : Trion campaign v3 (minimized via delta-debug)
Compiler   : onnxruntime, openvino, tensorflow, torch_compile, xla
Patterns   : TopK, Tile, Add, Relu, Sub, Neg, Abs, Relu...
Root cause : onnxruntime rel_L2=0.933
Minimal ops: TopK -> Tile -> Add -> Relu -> Sub -> Neg -> Abs -> Relu -> Mul -> Add -> Relu -> Reshape -> InstanceNormalization (13 ops, down from 14)
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
BACKENDS = ['onnxruntime', 'openvino', 'tensorflow', 'torch_compile', 'xla']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 128]
OUTPUT_NAME = 'n400_in1d_in'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([1]), 'n0_tk_k')
    i_1 = numpy_helper.from_array(np.array([1, 128], dtype=np.int64).reshape([2]), 'n0_tk_rep')
    i_2 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8A"
        "AAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAA"
        "AD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAA"
        "PwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/"
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8A"
        "AAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAA"
        "AD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAA"
        "PwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/"
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8="
    ), dtype=np.float32).copy().reshape([1, 128]), 'n200_anr_alpha')
    i_3 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    ), dtype=np.float32).copy().reshape([1, 128]), 'n300_rarelu_res')
    i_4 = numpy_helper.from_array(np.array([1, 1, 128], dtype=np.int64).reshape([3]), 'n400_in1d_s3d')
    i_5 = numpy_helper.from_array(np.array([1.0], dtype=np.float32).reshape([1]), 'n400_in1d_sc')
    i_6 = numpy_helper.from_array(np.array([0.0], dtype=np.float32).reshape([1]), 'n400_in1d_b')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6]


def _nodes():
    return [
        helper.make_node('TopK', inputs=['model_input', 'n0_tk_k'], outputs=['n0_tk_vals', 'n0_tk_idx'], axis=-1, largest=1, sorted=1),
        helper.make_node('Tile', inputs=['n0_tk_vals', 'n0_tk_rep'], outputs=['n0_tk_out']),
        helper.make_node('Add', inputs=['n0_tk_out', 'n0_tk_out'], outputs=['n100_ars_ad']),
        helper.make_node('Relu', inputs=['n100_ars_ad'], outputs=['n100_ars_rl']),
        helper.make_node('Sub', inputs=['n100_ars_rl', 'n0_tk_out'], outputs=['n100_ars_out']),
        helper.make_node('Neg', inputs=['n100_ars_out'], outputs=['n200_anr_neg']),
        helper.make_node('Abs', inputs=['n200_anr_neg'], outputs=['n200_anr_abs']),
        helper.make_node('Relu', inputs=['n200_anr_abs'], outputs=['n200_anr_relu']),
        helper.make_node('Mul', inputs=['n200_anr_relu', 'n200_anr_alpha'], outputs=['n200_anr_out']),
        helper.make_node('Add', inputs=['n200_anr_out', 'n300_rarelu_res'], outputs=['n300_rarelu_add']),
        helper.make_node('Relu', inputs=['n300_rarelu_add'], outputs=['n300_rarelu_out']),
        helper.make_node('Reshape', inputs=['n300_rarelu_out', 'n400_in1d_s3d'], outputs=['n400_in1d_r3']),
        helper.make_node('InstanceNormalization', inputs=['n400_in1d_r3', 'n400_in1d_sc', 'n400_in1d_b'], outputs=['n400_in1d_in'], epsilon=9.999999747378752e-06),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000308_min", inputs_info, outputs_info, initializer=_inits())
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = IR_VERSION
    return m.SerializeToString()


def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "z5Phv72Fuz4WLB4+Rg23v5zxfD7tTFU/KfnevrMAFT92RYs/c4bNvxwG+j+mKy6/DqJGvzNmoL+w"
        "org+Q0HTvpuR4D4BF28+opdpvtxSQ7/0K32+BXZ/POTsWr+CX2+/FJ5fvT/BNj+mjB6/X/Olv3Rb"
        "ML8osPW9MRz3P1BhI79526y/IRwUP5JLyT7vl2e/MrZgP2wQVz8GqLg+RdcEP/v29j4Nr5M/KFmV"
        "vj7xK7+29cQ/4zhXP1smLD3dN7W+ZkBMPz1+Ur/Yy9i+FNfFPphwXL6bSZc/FFfIvNRRbz+a9L+/"
        "H+x4P/65eD1Va6C+N2yrPlKgNz/47r4/sNizvHlSmT/8rlG+kNCBPrT8ur4K1XE+stZ0PxDU2r8q"
        "vNM+yP/2veQ4qT0N7mK/NUUpv1+TlT/n+Fo/Az1/P4zRYr82SKc+SaO9v8HUJD9Kejs+XEzXP/za"
        "MD/4Ja+/FhgaP+8e3z68uJ++AdjOvzqvDT524G4+DJiGvwbkBD6KRmO+OyeFP8KbmD/zQA7A5Hk2"
        "PzeJ6T5iSZs/Qri6P/VpLT9ZoSE+vjPLvbvhq77OdAy/bzllP68nSb97GDK/oULQvQKN1T6tIQ3A"
        "sh2Bvy6ytb/jrVq/f4QxwMCbrL+a8k4/mK0Vvnpg8r7wC3K/A3HyvopKD7/NFjJAdl24PxPj0b8="
    ), dtype=np.float32).copy().reshape([1, 128])


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
