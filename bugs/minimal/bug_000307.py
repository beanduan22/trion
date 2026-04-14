#!/usr/bin/env python3
"""
Bug ID     : bug_000307
Source     : Trion campaign v3 (fuzzing)
Compiler   : tensorflow
Patterns   : reshape-flatten + BN + LeakyRelu
Root cause : reflect-Pad + Reshape-flatten + Conv(s=2) + BN + LeakyRelu + MatMul + Relu+Add+Relu
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
OUTPUT_NAME = 'n400_rar_out'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_SHAPE = [1, 64, 17, 17]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.array([0, 0, 2, 0, 0, 0, 0, 2], dtype=np.int64).reshape([8]), 'n0_prma_pads')
    i_1 = numpy_helper.from_array(np.array([1, -1], dtype=np.int64).reshape([2]), 'n100_rfr_fl')
    i_2 = numpy_helper.from_array(np.array([1, 3, 34, 34], dtype=np.int64).reshape([4]), 'n100_rfr_bk')
    i_3 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "B7WLPxQYhL+0lqm6+i61P6os5D6vaAK/XtkmP4BjzzzaPXE+mjgevhdFGD5LMnm/z6wqvxMpFD4U"
        "Qhm/JygiP0Opdb/dY5C/6OyrvrXekz92pZw+CsDNPtLDhT8mM9+/41gzvhIZYT/Yx+w882NRv5ae"
        "1T8jxQ0/OQiVP4sepz6sShe+GuMiPwWNmj9FYiC/WOGZPzgoPT/rooY+mRCSv2Stu776LbU9agor"
        "v01Cjr/JcQa/ECoAv/rvb7+NzVI+JaRMv7i0RD74VTW+A6TFvsf5mj+jexi+TncSvX1iQb5ikLI+"
        "GzxpPyXAQz5DSbA/wjNwP6iQAb9bk/M9L/rbvfMkkD5LJ2W/Nxw3QDYptb6XJ+Q+mYLhvg0L4z6b"
        "njS+edr7v4w5Sr6mJLY/IwGov0xWoT9fUry+5eNLPpb70D3iKjE/q6wPPkUrDz8WULg9a/sVvyb5"
        "cz8eO+O+CrkKP2zgOL++of0+eaXEPujwlT7TOS+/f0Rpvy0uOj/o9+s9Ax8pP2HFjL7HISq+ARKN"
        "v7Arnr87nYe/CbX0vsUrQj//kAO/AI6hvVgrhb5+Cj2/vxcGvyev9j7F9le/KJGDvhVTVD+jYiq/"
        "aFuGPlEAP70b1zu9wjaEPypt0j939uu7JNM2P3IHuT5prYi9gbssv249vb6G6S4/a1Icv7g5j75Y"
        "jzu/RiQEvmiPgb9AM629bfBJPxW3FT6P8JC/bpwQPdx3Kb/jUVY/jlPYPtI0iL6Su2Q/Yak0P3kG"
        "fT4D+ui/Cl3NPf7Smr7ls6Y/oHmcvzcZSb+mXWK/VP6PP6uJRL7/M72+Wi8DPxV9Vb+4R4G/tWbj"
        "vjAwyr0KYl2+H0iSP0QXmT928kS+uES2vr3vJr6VAKS/tAyFPsUsVD/m/AE/0HdQPh50kj1lXPA9"
        "W2uMPnnGA79t1Oq+iqXsPm5lfr8hMZe/0yxOP+sp7r0+ur4/u4lTvtJvSb5ULBi/Qm5ovb+St799"
        "f4k/P9kJvhZnyz4gS98+8RMRPyCdQ70jloi+"
    ), dtype=np.float32).copy().reshape([64, 3, 1, 1]), 'n200_cblrelu_w')
    i_4 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    ), dtype=np.float32).copy().reshape([64]), 'n200_cblrelu_b')
    i_5 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPw=="
    ), dtype=np.float32).copy().reshape([64]), 'n200_cblrelu_bn_scale')
    i_6 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    ), dtype=np.float32).copy().reshape([64]), 'n200_cblrelu_bn_bias')
    i_7 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    ), dtype=np.float32).copy().reshape([64]), 'n200_cblrelu_bn_mean')
    i_8 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPw=="
    ), dtype=np.float32).copy().reshape([64]), 'n200_cblrelu_bn_var')
    i_9 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "X68Xv/Eytz7S7RA9OGCgvvX8qD4eRN8+wghePjMO+r7liJe+fQVfPj1lk77EIhw+O2Q/vrwTMr7k"
        "wca+9ciYvhYltr7yNty8K+gIvm4izb0gMaM9zMxRPoZzyj6YcJs+AGV6vlSqAj4lXha/a/WyvTyH"
        "Xz7iG+W+BhkyvZ+Knbua2o0+BNgrvgxACD5ZG5g+BybLPiIh/L50PiA+CE6nPDce1j5EGA6+fAEH"
        "PgfCP77yPSK9t/BNPvnlNL46rpy+9+POviQWAj//Jrm+xK16vU7GXTzNR3k+UY2mvrDT+D6QXSE/"
        "P0u5vrnlAD5zXQ4/DyUbvuwdI7768qM+qFcJPiQH9L01niG/41YOPj6O8z65nao8hISGPuVFqb5x"
        "BcA+M2Wuvk0eGz9bMEc8BBdvvquggL72zqm+UOOSPiYLmj5zeLC+YFWyvWUxkj7pzmA8h/nKPn6X"
        "wj7/jtm+AgswPq7Qtr2Iz0Q9XHd4vnci1b2gZoS9eVh6PkD7yL5mw6y+Oq9dPtioFj/GAA2/7I67"
        "vQg+qL4U6vo6fdSKvmtTj76AAKm+whqPPgA8rr4fgEG+tqUBvnBphL4eAbY+V9ATP+wqkzxP3cQ9"
        "kVPAvqT0BL5kFSg+vXHoPB17Mr5Qqku/OD9Av0Gdqr0j20a/qrhOPz39077GYPu9g4rjvIUzAj+b"
        "yom8ilpdvneM1D3vwOE+lKTjPLe11D7fE8E96YHEvBY4lD7fhXM+yHviPbWaOz/6fLI+4eqcPrku"
        "sj7zYtU+d0G3vmLEgj7MGik//U23vm5Pjb0HMuI9l0YBvwGfRL72wTk/+0dbP5rrIL4M97O9yFaW"
        "PtlKib4Jth6/khIJP3erub6xjwU9+csJP3SCkD4e6ae+gQ4gPpSOzD46oU4+jTsTPYD9Ub4DswE+"
        "EXbTPqD/0j7nxai+CSKQvr7hmr67VvM9JYrEvmPLrL6pV3M+vHZsvt4oYL4KqWC+/LnIPs+fvDuw"
        "nK+8NRpKvYP/Cb+Uwt29kpgJPWGg5r0S7da+ambkvX9XmL4vCxQ9AYmMPMOZK73L7gk+dtMKPvql"
        "8j2bW6i9pMTRvhJtAj+vteG+YRqZPjPKgD4RAou+qpS4vq7IwTweeN8+Px62vnwV1z3eDSo7Q9C7"
        "Pjyshj4yTzm+Z587vhdR5L52OgA+mr6lvZ9q7DvSVTy+9f0HP0qUlz6pJay+JZUGv46zFj/eyuG+"
        "uVw7vt87/j5Q0/I9fVhYv2b5gzs1uIi+t9bRPebVvD5y36e+HoiqvVMbiD5RFMu+tuk5P5Q0U77l"
        "bpi8GHT7PK0oxT72l9U+JiyFPXAjNb5VVQm86OGqPZQayr0IgY2+/ua+vu0Ugr6Dbr6+jhjTvsT/"
        "Cb2+Mh0+2HW9vgbtlT60v6o+I6xSPwyFwz0V1uu9xXJnvj750j4wu6c9guAnPz5jGb6G94G+3Xwd"
        "Pwiq7rwhIV6+Sbq4vtQ3kT6FAg6/tcNcPVIlsb1fqec+pNYuv4g2LT5QUlw+InimPoQLcT7+bYy+"
        "upCMPhJgHz3rCse+jNKEPg=="
    ), dtype=np.float32).copy().reshape([1, 1, 17, 17]), 'n300_mm4d_w')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('Pad', inputs=['model_input', 'n0_prma_pads'], outputs=['n0_prma_out'], mode='reflect'),
        helper.make_node('Reshape', inputs=['n0_prma_out', 'n100_rfr_fl'], outputs=['n100_rfr_r1']),
        helper.make_node('Reshape', inputs=['n100_rfr_r1', 'n100_rfr_bk'], outputs=['n100_rfr_out']),
        helper.make_node('Conv', inputs=['n100_rfr_out', 'n200_cblrelu_w', 'n200_cblrelu_b'], outputs=['n200_cblrelu_cv'], kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[2, 2]),
        helper.make_node('BatchNormalization', inputs=['n200_cblrelu_cv', 'n200_cblrelu_bn_scale', 'n200_cblrelu_bn_bias', 'n200_cblrelu_bn_mean', 'n200_cblrelu_bn_var'], outputs=['n200_cblrelu_bn'], epsilon=9.999999747378752e-06),
        helper.make_node('LeakyRelu', inputs=['n200_cblrelu_bn'], outputs=['n200_cblrelu_out'], alpha=0.10000000149011612),
        helper.make_node('MatMul', inputs=['n200_cblrelu_out', 'n300_mm4d_w'], outputs=['n300_mm4d_out']),
        helper.make_node('Relu', inputs=['n300_mm4d_out'], outputs=['n400_rar_r1']),
        helper.make_node('Add', inputs=['n400_rar_r1', 'n300_mm4d_out'], outputs=['n400_rar_ad']),
        helper.make_node('Relu', inputs=['n400_rar_ad'], outputs=['n400_rar_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000307", inputs_info, outputs_info, initializer=_inits())
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
