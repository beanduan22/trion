#!/usr/bin/env python3
"""
Bug ID     : bug_000267
Source     : Trion campaign v3 (fuzzing)
Compiler   : onnxruntime, openvino, tensorflow, xla
Patterns   : Tanh+Erf+CumSum+Pow(1)
Root cause : matmul-bias-sigmoid+Tanh+Erf+CumSum+Pow(x,1) identity
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
OPSET = 17
IR_VERSION = 8
INPUT_NAME = 'model_input'
OUTPUT_NAME = 'n400_pio_out'
INPUT_SHAPE = [1, 64]
OUTPUT_SHAPE = [1, 64]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
        "PFeC/b21LeY9nfWgvKrKKb3NjkQ+PRWXPA=="
    ), dtype=np.float32).copy().reshape([64, 64]), 'n0_mmbsig_w')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    ), dtype=np.float32).copy().reshape([64]), 'n0_mmbsig_b')
    i_2 = numpy_helper.from_array(np.array([0.10000000149011612], dtype=np.float32).reshape([1]), 'n100_ama_a')
    i_3 = numpy_helper.from_array(np.array([1.5], dtype=np.float32).reshape([1]), 'n100_ama_b')
    i_4 = numpy_helper.from_array(np.array([0.20000000298023224], dtype=np.float32).reshape([1]), 'n100_ama_c')
    i_5 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([]), 'n300_cs_ax')
    i_6 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([1]), 'n400_pio_one')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('MatMul', inputs=['model_input', 'n0_mmbsig_w'], outputs=['n0_mmbsig_mm']),
        helper.make_node('Add', inputs=['n0_mmbsig_mm', 'n0_mmbsig_b'], outputs=['n0_mmbsig_add']),
        helper.make_node('Sigmoid', inputs=['n0_mmbsig_add'], outputs=['n0_mmbsig_out']),
        helper.make_node('Add', inputs=['n0_mmbsig_out', 'n100_ama_a'], outputs=['n100_ama_x1']),
        helper.make_node('Mul', inputs=['n100_ama_x1', 'n100_ama_b'], outputs=['n100_ama_x2']),
        helper.make_node('Add', inputs=['n100_ama_x2', 'n100_ama_c'], outputs=['n100_ama_out']),
        helper.make_node('Tanh', inputs=['n100_ama_out'], outputs=['n200_erf__th']),
        helper.make_node('Erf', inputs=['n200_erf__th'], outputs=['n200_erf__act']),
        helper.make_node('Mul', inputs=['n200_erf__act', 'n100_ama_out'], outputs=['n200_erf__out']),
        helper.make_node('CumSum', inputs=['n200_erf__out', 'n300_cs_ax'], outputs=['n300_cs_out']),
        helper.make_node('Pow', inputs=['n300_cs_out', 'n400_pio_one'], outputs=['n400_pio_pw']),
        helper.make_node('Add', inputs=['n400_pio_pw', 'n300_cs_out'], outputs=['n400_pio_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000267", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "7XKhvsoDE75gs5G+VTxDP7C2AMCxm9y+dJcQwCRSIz/8afc/2S5PPmgNmr9qS0u/2pYUvzJ4Y7+d"
        "aNA/XrXlPhx9iT5xxQdA5NYkP9+5aT4SaJk/XSJbP5ibi79+tkM+iy0LPx/IGT6/9me+EQ3Ivncc"
        "rz9QSgVAa66zPn7QkbySv8c+EJQHvtMzH71wQBE/QZeUP8/jjz+fCuy/4PSiPxDTjL65KnM/2OoY"
        "P2p08z7500m/PbZzP68GOj/gJuO/mDyevdXjNT9Cpfa+nzEyvohtFECBlV0/CLUtP1xJjr9v+J+/"
        "+yOEvzJ1nD8FR4w/FtcZPwWpDT+jMNI+51gHvw=="
    ), dtype=np.float32).copy().reshape([1, 64])


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
