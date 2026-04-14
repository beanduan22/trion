#!/usr/bin/env python3
"""
Bug ID     : bug_000163
Source     : Trion campaign v3 (fuzzing)
Compiler   : onnxruntime, openvino, tensorflow, xla
Patterns   : GLU + LogSoftmax + CumSum + cast
Root cause : GLU Split + LogSoftmax + CumSum + fp32↔int32 cast roundtrip
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
OUTPUT_NAME = 'n400_cfir_out'
INPUT_SHAPE = [1, 16, 64]
OUTPUT_SHAPE = [1, 16, 32]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.array([32, 32], dtype=np.int64).reshape([2]), 'n0_glu_sp')
    i_1 = numpy_helper.from_array(np.array([2], dtype=np.int64).reshape([]), 'n200_cs_ax')
    i_2 = numpy_helper.from_array(np.array([1.2000000476837158], dtype=np.float32).reshape([1]), 'n300_mam_a')
    i_3 = numpy_helper.from_array(np.array([0.30000001192092896], dtype=np.float32).reshape([1]), 'n300_mam_b')
    i_4 = numpy_helper.from_array(np.array([1.5], dtype=np.float32).reshape([1]), 'n300_mam_c')
    return [i_0, i_1, i_2, i_3, i_4]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('Split', inputs=['model_input', 'n0_glu_sp'], outputs=['n0_glu_a', 'n0_glu_b'], axis=-1),
        helper.make_node('Sigmoid', inputs=['n0_glu_b'], outputs=['n0_glu_sig']),
        helper.make_node('Mul', inputs=['n0_glu_a', 'n0_glu_sig'], outputs=['n0_glu_out']),
        helper.make_node('LogSoftmax', inputs=['n0_glu_out'], outputs=['n100_lsm_ls'], axis=-1),
        helper.make_node('Mul', inputs=['n100_lsm_ls', 'n0_glu_out'], outputs=['n100_lsm_out']),
        helper.make_node('CumSum', inputs=['n100_lsm_out', 'n200_cs_ax'], outputs=['n200_cs_out']),
        helper.make_node('Mul', inputs=['n200_cs_out', 'n300_mam_a'], outputs=['n300_mam_x1']),
        helper.make_node('Add', inputs=['n300_mam_x1', 'n300_mam_b'], outputs=['n300_mam_x2']),
        helper.make_node('Mul', inputs=['n300_mam_x2', 'n300_mam_c'], outputs=['n300_mam_out']),
        helper.make_node('Abs', inputs=['n300_mam_out'], outputs=['n400_cfir_ab']),
        helper.make_node('Ceil', inputs=['n400_cfir_ab'], outputs=['n400_cfir_ce']),
        helper.make_node('Cast', inputs=['n400_cfir_ce'], outputs=['n400_cfir_ci'], to=6),
        helper.make_node('Cast', inputs=['n400_cfir_ci'], outputs=['n400_cfir_cf'], to=1),
        helper.make_node('Mul', inputs=['n400_cfir_cf', 'n300_mam_out'], outputs=['n400_cfir_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000163", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 16, 64])


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
