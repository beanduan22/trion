#!/usr/bin/env python3
"""
Bug ID     : bug_000308
Source     : Trion campaign v3 (fuzzing)
Compiler   : onnxruntime, openvino, tensorflow, torch_compile, xla
Patterns   : TopK+Tile + InstanceNorm
Root cause : TopK(k=1)+Tile + sign-manipulation + InstanceNorm1D (5 backends)
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
OPSET = 17
IR_VERSION = 8
INPUT_NAME = 'model_input'
OUTPUT_NAME = 'n400_in1d_out'
INPUT_SHAPE = [1, 128]
OUTPUT_SHAPE = [1, 128]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
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
    i_5 = numpy_helper.from_array(np.array([1, 128], dtype=np.int64).reshape([2]), 'n400_in1d_s2d')
    i_6 = numpy_helper.from_array(np.array([1.0], dtype=np.float32).reshape([1]), 'n400_in1d_sc')
    i_7 = numpy_helper.from_array(np.array([0.0], dtype=np.float32).reshape([1]), 'n400_in1d_b')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
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
        helper.make_node('Reshape', inputs=['n400_in1d_in', 'n400_in1d_s2d'], outputs=['n400_in1d_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000308", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Reference: pure PyTorch eager via onnx2torch
# ------------------------------------------------------------------
def _ref_pytorch(model_bytes: bytes, x: np.ndarray) -> np.ndarray:
    import torch, onnx2torch
    m = onnx2torch.convert(onnx.load_from_string(model_bytes)).eval()
    with torch.no_grad():
        out = m(torch.from_numpy(x))
    if isinstance(out, (list, tuple)): out = out[0]
    return out.detach().cpu().float().numpy().ravel()


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-12))


# ------------------------------------------------------------------
# Backend drivers
# ------------------------------------------------------------------
def _run_onnxruntime(model_bytes, x):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(model_bytes, opts, providers=["CPUExecutionProvider"])
        return np.asarray(sess.run(None, {INPUT_NAME: x})[0]).ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_openvino(model_bytes, x):
    try:
        import openvino as ov
        core = ov.Core()
        compiled = core.compile_model(core.read_model(io.BytesIO(model_bytes)), "CPU")
        out = compiled([x])
        return np.asarray(list(out.values())[0]).ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_tensorflow(model_bytes, x, *, jit=False):
    try:
        sys.path.insert(0, "/home/binduan/myspace/trion")
        from trion.oracle.tf_backend import TFBackend
        r = TFBackend().run(onnx.load_from_string(model_bytes), {INPUT_NAME: x}, optimized=jit)
        if r.output is None: return None, (r.error or "run returned None")[:120]
        return r.output.ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_xla(model_bytes, x):
    return _run_tensorflow(model_bytes, x, jit=True)


def _run_torchscript(model_bytes, x):
    try:
        import torch, onnx2torch
        m = onnx2torch.convert(onnx.load_from_string(model_bytes)).eval()
        t = torch.from_numpy(x)
        scripted = torch.jit.trace(m, (t,))
        frozen = torch.jit.optimize_for_inference(torch.jit.freeze(scripted))
        with torch.no_grad():
            out = frozen(t)
        if isinstance(out, (list, tuple)): out = out[0]
        return out.detach().cpu().float().numpy().ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_torch_compile(model_bytes, x):
    try:
        import torch, onnx2torch
        m = onnx2torch.convert(onnx.load_from_string(model_bytes)).eval()
        compiled = torch.compile(m, mode="reduce-overhead", fullgraph=False)
        with torch.no_grad():
            out = compiled(torch.from_numpy(x))
        if isinstance(out, (list, tuple)): out = out[0]
        return out.detach().cpu().float().numpy().ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


RUNNERS = {
    "onnxruntime":   _run_onnxruntime,
    "openvino":      _run_openvino,
    "tensorflow":    _run_tensorflow,
    "xla":           _run_xla,
    "torchscript":   _run_torchscript,
    "torch_compile": _run_torch_compile,
}


def main() -> int:
    try:
        model_bytes = build_model()
        x = _input()
        ref = _ref_pytorch(model_bytes, x)
    except Exception as e:
        print(f"setup failed: {type(e).__name__}: {e}"); return 2

    print(f"Bug ID:    {__doc__.splitlines()[2].split(':',1)[1].strip()}")
    print(f"Backends:  {BACKENDS}")
    print(f"Tolerance: {TOLERANCE}")

    any_bug = False
    for backend in BACKENDS:
        run = RUNNERS.get(backend)
        if run is None:
            print(f"  [{backend}] no driver"); continue
        out, err = run(model_bytes, x)
        if err is not None:
            print(f"  [{backend}] CRASH: {err}   →   BUG REPRODUCED")
            any_bug = True
            continue
        diff = _rel_l2(out, ref)
        verdict = "REPRODUCED" if diff > TOLERANCE else "ok"
        print(f"  [{backend}] rel_L2 vs pytorch_ref = {diff:.4e}   →   {verdict}")
        if diff > TOLERANCE:
            any_bug = True

    PASS = not any_bug
    print(f"PASS={PASS}")
    if not PASS:
        print("BUG REPRODUCED"); return 0
    print("not reproduced"); return 1


if __name__ == "__main__":
    sys.exit(main())
