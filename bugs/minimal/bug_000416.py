#!/usr/bin/env python3
"""
Bug ID     : bug_000416
Source     : Trion campaign v3 (fuzzing)
Compiler   : tensorflow
Patterns   : Transpose squash + ReduceMean axis=0
Root cause : Two Transpose[2,3,0,1]+ReduceL2+Div+Mul+reflect-Pad+Conv+Pow+ReduceMean(axis=0)
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
OPSET = 17
IR_VERSION = 8
INPUT_NAME = 'model_input'
OUTPUT_NAME = 'n400_rmf_out'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_SHAPE = [1, 32, 32, 32]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.array([9.999999960041972e-13], dtype=np.float32).reshape([1]), 'n100_l2n2_eps')
    i_1 = numpy_helper.from_array(np.array([1.0], dtype=np.float32).reshape([1, 1, 1, 1]), 'n100_l2n2_scale')
    i_2 = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64).reshape([8]), 'n200_prc_pads')
    i_3 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([32, 3, 3, 3]), 'n200_prc_w')
    i_4 = numpy_helper.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape([32]), 'n200_prc_b')
    i_5 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([1]), 'n300_pio_one')
    return [i_0, i_1, i_2, i_3, i_4, i_5]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('Transpose', inputs=['model_input'], outputs=['n0_tts_t1'], perm=[2, 3, 0, 1]),
        helper.make_node('Transpose', inputs=['n0_tts_t1'], outputs=['n0_tts_out'], perm=[2, 3, 0, 1]),
        helper.make_node('ReduceL2', inputs=['n0_tts_out'], outputs=['n100_l2n2_norm'], axes=[-1], keepdims=1),
        helper.make_node('Add', inputs=['n100_l2n2_norm', 'n100_l2n2_eps'], outputs=['n100_l2n2_add']),
        helper.make_node('Div', inputs=['n0_tts_out', 'n100_l2n2_add'], outputs=['n100_l2n2_div']),
        helper.make_node('Mul', inputs=['n100_l2n2_div', 'n100_l2n2_scale'], outputs=['n100_l2n2_out']),
        helper.make_node('Pad', inputs=['n100_l2n2_out', 'n200_prc_pads'], outputs=['n200_prc_pad'], mode='reflect'),
        helper.make_node('Conv', inputs=['n200_prc_pad', 'n200_prc_w', 'n200_prc_b'], outputs=['n200_prc_out'], kernel_shape=[3, 3], pads=[0, 0, 0, 0]),
        helper.make_node('Pow', inputs=['n200_prc_out', 'n300_pio_one'], outputs=['n300_pio_pw']),
        helper.make_node('Add', inputs=['n300_pio_pw', 'n200_prc_out'], outputs=['n300_pio_out']),
        helper.make_node('ReduceMean', inputs=['n300_pio_out'], outputs=['n400_rmf_red'], axes=[0], keepdims=1),
        helper.make_node('Add', inputs=['n300_pio_out', 'n400_rmf_red'], outputs=['n400_rmf_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000416", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 3, 32, 32])


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
