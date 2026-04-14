#!/usr/bin/env python3
"""
Bug ID     : bug_000143
Source     : Trion campaign v3 (fuzzing)
Compiler   : onnxruntime, tensorflow
Patterns   : Neg+Abs+Relu + ReduceMax + manual LayerNorm
Root cause : Neg+Abs+Relu sign chain → ReduceMax → gated residual → manual LayerNorm
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
BACKENDS = ['onnxruntime', 'tensorflow']
OPSET = 17
IR_VERSION = 8
INPUT_NAME = 'model_input'
OUTPUT_NAME = 'n400_mln_out'
INPUT_SHAPE = [1, 128]
OUTPUT_SHAPE = [1, 128]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.array([0.19801978766918182], dtype=np.float32).reshape([1]), 'n0_mbr_rc')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    ), dtype=np.float32).copy().reshape([1, 128]), 'n0_mbr_bias')
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
    ), dtype=np.float32).copy().reshape([1, 128]), 'n100_anr_alpha')
    i_3 = numpy_helper.from_array(np.array([0.0], dtype=np.float32).reshape([1]), 'n200_redu_zero')
    i_4 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8A"
        "AAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAA"
        "AD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAA"
        "PwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/"
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8A"
        "AAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAA"
        "AD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAA"
        "PwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/"
        "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAD8="
    ), dtype=np.float32).copy().reshape([1, 128]), 'n300_gr_wg')
    i_5 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8="
    ), dtype=np.float32).copy().reshape([1, 128]), 'n300_gr_one')
    i_6 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9m"
        "ZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zm"
        "hj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaG"
        "P2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/"
        "ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9m"
        "ZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zm"
        "hj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaG"
        "P2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/"
        "ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj9mZoY/ZmaGP2Zmhj8="
    ), dtype=np.float32).copy().reshape([1, 128]), 'n300_gr_scale')
    i_7 = numpy_helper.from_array(np.array([9.999999747378752e-06], dtype=np.float32).reshape([1]), 'n400_mln_eps')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('Mul', inputs=['model_input', 'n0_mbr_rc'], outputs=['n0_mbr_mul']),
        helper.make_node('Add', inputs=['n0_mbr_mul', 'n0_mbr_bias'], outputs=['n0_mbr_out']),
        helper.make_node('Neg', inputs=['n0_mbr_out'], outputs=['n100_anr_neg']),
        helper.make_node('Abs', inputs=['n100_anr_neg'], outputs=['n100_anr_abs']),
        helper.make_node('Relu', inputs=['n100_anr_abs'], outputs=['n100_anr_relu']),
        helper.make_node('Mul', inputs=['n100_anr_relu', 'n100_anr_alpha'], outputs=['n100_anr_out']),
        helper.make_node('ReduceMax', inputs=['n100_anr_out'], outputs=['n200_redu_red'], axes=[-1], keepdims=1),
        helper.make_node('Mul', inputs=['n100_anr_out', 'n200_redu_zero'], outputs=['n200_redu_z']),
        helper.make_node('Add', inputs=['n200_redu_z', 'n200_redu_red'], outputs=['n200_redu_out']),
        helper.make_node('Mul', inputs=['n200_redu_out', 'n300_gr_scale'], outputs=['n300_gr_sc']),
        helper.make_node('Sigmoid', inputs=['n300_gr_wg'], outputs=['n300_gr_gate']),
        helper.make_node('Sub', inputs=['n300_gr_one', 'n300_gr_gate'], outputs=['n300_gr_inv']),
        helper.make_node('Mul', inputs=['n300_gr_sc', 'n300_gr_gate'], outputs=['n300_gr_lhs']),
        helper.make_node('Mul', inputs=['n200_redu_out', 'n300_gr_inv'], outputs=['n300_gr_rhs']),
        helper.make_node('Add', inputs=['n300_gr_lhs', 'n300_gr_rhs'], outputs=['n300_gr_out']),
        helper.make_node('ReduceMean', inputs=['n300_gr_out'], outputs=['n400_mln_m'], axes=[-1], keepdims=1),
        helper.make_node('Sub', inputs=['n300_gr_out', 'n400_mln_m'], outputs=['n400_mln_sub']),
        helper.make_node('Mul', inputs=['n400_mln_sub', 'n400_mln_sub'], outputs=['n400_mln_sq']),
        helper.make_node('ReduceMean', inputs=['n400_mln_sq'], outputs=['n400_mln_var'], axes=[-1], keepdims=1),
        helper.make_node('Add', inputs=['n400_mln_var', 'n400_mln_eps'], outputs=['n400_mln_add']),
        helper.make_node('Sqrt', inputs=['n400_mln_add'], outputs=['n400_mln_sqrt']),
        helper.make_node('Div', inputs=['n400_mln_sub', 'n400_mln_sqrt'], outputs=['n400_mln_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000143", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "GHuEvWXwTkDuk6Q+YbHEvxZkFT8mtmS/1hUYvxySU77TUry+lz+3PyXS/r48na68LUkLP9E/bL6m"
        "mS8+EU+av/bXU70YXBE/tIdnvsvMNr+TZ6E/xvvEPkeH+T5IvfS+1C2Ev8aNZD+4pq2/bOvePJm4"
        "Sz/ynme/+CAfP5Fp/j82iI++acfNPyX6wb1rfS6/hp2HP9YQH7+TYZc9WLIPP3ODK798RJk+5XY8"
        "P9iU6r4B2Ka/KYVmP4Mmfj7px2K+B+0rP1RiDL3WuRtAeH24vYqGgL06p7q/2O2Mv3Q/CT9aOIw/"
        "QAQivlFyxz5/oQI/vKFnPwXLBj+YMam/xWwLQKyPnL4D4Yu+kzDlvqLYl7yB2tY/dSWHv3EiNb9n"
        "fXu/E5NDPyF2lz4hZDE/NWePv1ywpD6ojq4/acKyP+3ZHT/fZtM+7kctv/u7az+MXVy/IfIXP5le"
        "/r/b7iW/e9aAPo0rZr9Psqq//EIZwDVpxj+3Kw+/cAbYP6NGWj/3lX4/1+hOv0KDnz+Vnuw/Z3mf"
        "PzrSpb+PeU4/tHcnv/kpeL9eLI8+UI9SPwNynL9PJ9o8RMzPPqIII8BM+PE+xp7vvpQ0kD2p1Qw+"
        "On1ev1+0I7/BeNG+FocwP3kltD6q4829xrhPP2jE9L88Oli/T/EtP0X47b4mXGQ+qfN/PxWwo78="
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
