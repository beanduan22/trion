#!/usr/bin/env python3
"""
Bug ID     : bug_000060
Source     : Trion campaign v3 (fuzzing)
Compiler   : onnxruntime, openvino, tensorflow, xla
Patterns   : CumSum + SiLU residual
Root cause : CumSum(-1)+SiLU residual+gated residual+Gather/Slice/Expand
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
OUTPUT_NAME = 'n400_sigs_out'
INPUT_SHAPE = [1, 64]
OUTPUT_SHAPE = [1, 64]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([]), 'n0_cs_ax')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8A"
        "AAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAA"
        "AD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAA"
        "PwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/"
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPw=="
    ), dtype=np.float32).copy().reshape([1, 64]), 'n200_gr_wg')
    i_2 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPw=="
    ), dtype=np.float32).copy().reshape([1, 64]), 'n200_gr_one')
    i_3 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9m"
        "ZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zm"
        "hj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaG"
        "P2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/"
        "ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGPw=="
    ), dtype=np.float32).copy().reshape([1, 64]), 'n200_gr_scale')
    i_4 = numpy_helper.from_array(np.array([0], dtype=np.int64).reshape([1]), 'n300_gsc_idx')
    i_5 = numpy_helper.from_array(np.array([0], dtype=np.int64).reshape([1]), 'n300_gsc_st')
    i_6 = numpy_helper.from_array(np.array([64], dtype=np.int64).reshape([1]), 'n300_gsc_en')
    i_7 = numpy_helper.from_array(np.array([1, 64], dtype=np.int64).reshape([2]), 'n300_gsc_exp')
    i_8 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAA"
        "AABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAA"
        "AEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAA"
        "QAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABA"
        "AAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQA=="
    ), dtype=np.float32).copy().reshape([1, 64]), 'n400_sigs_two')
    i_9 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPw=="
    ), dtype=np.float32).copy().reshape([1, 64]), 'n400_sigs_one')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('CumSum', inputs=['model_input', 'n0_cs_ax'], outputs=['n0_cs_out']),
        helper.make_node('Sigmoid', inputs=['n0_cs_out'], outputs=['n100_sir_sg']),
        helper.make_node('Mul', inputs=['n0_cs_out', 'n100_sir_sg'], outputs=['n100_sir_ml']),
        helper.make_node('Add', inputs=['n100_sir_ml', 'n0_cs_out'], outputs=['n100_sir_out']),
        helper.make_node('Mul', inputs=['n100_sir_out', 'n200_gr_scale'], outputs=['n200_gr_sc']),
        helper.make_node('Sigmoid', inputs=['n200_gr_wg'], outputs=['n200_gr_gate']),
        helper.make_node('Sub', inputs=['n200_gr_one', 'n200_gr_gate'], outputs=['n200_gr_inv']),
        helper.make_node('Mul', inputs=['n200_gr_sc', 'n200_gr_gate'], outputs=['n200_gr_lhs']),
        helper.make_node('Mul', inputs=['n100_sir_out', 'n200_gr_inv'], outputs=['n200_gr_rhs']),
        helper.make_node('Add', inputs=['n200_gr_lhs', 'n200_gr_rhs'], outputs=['n200_gr_out']),
        helper.make_node('Gather', inputs=['n200_gr_out', 'n300_gsc_idx'], outputs=['n300_gsc_g'], axis=0),
        helper.make_node('Slice', inputs=['n300_gsc_g', 'n300_gsc_st', 'n300_gsc_en'], outputs=['n300_gsc_sl']),
        helper.make_node('Expand', inputs=['n300_gsc_sl', 'n300_gsc_exp'], outputs=['n300_gsc_out']),
        helper.make_node('Sigmoid', inputs=['n300_gsc_out'], outputs=['n400_sigs_sig']),
        helper.make_node('Mul', inputs=['n400_sigs_sig', 'n400_sigs_two'], outputs=['n400_sigs_sc']),
        helper.make_node('Sub', inputs=['n400_sigs_sc', 'n400_sigs_one'], outputs=['n400_sigs_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000060", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "nP7EPzaMEsBiQ0o9fsGBv+9eI75UMdg8IB6wPv/QJL9dQ2c/WnkfPgwBQL2CbGA//OHRv0LdhT++"
        "uBG+Dgabv7ve1L8+xyM/B+IBv1h3tL1d84i/xm8OPupDpb/t4W0/YsOyP43oe79/ZKe+f9xWvX7A"
        "pb4hZuK+m4kHP9k9gT+Pjcq/IyGOPsbhIr+jhrq/ipKtvAAqKL/YUda/SRojv/8N1r8r/WQ/5eXg"
        "vtLlfUAyrIe/mCkBv40W6r5txYu+kyN2v5N3aj1j5X+/RXg3vDga/D1mRK+/EInXvpYkPj+/szO+"
        "yaifvvybnDxwmbS/hVOCPkXQur5xnc0/7H72vg=="
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
