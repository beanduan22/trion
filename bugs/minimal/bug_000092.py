#!/usr/bin/env python3
"""
Bug ID     : bug_000092
Source     : Trion campaign v3 (fuzzing)
Compiler   : onnxruntime, openvino, tensorflow, xla
Patterns   : cRMS-norm + CumSum
Root cause : Manual cRMS-norm + ReLU residual + CumSum + Gather
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
OUTPUT_NAME = 'n400_gsc_out'
INPUT_SHAPE = [1, 256]
OUTPUT_SHAPE = [1, 256]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPw=="
    ), dtype=np.float32).copy().reshape([256]), 'n0_crms_g')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "/puBPz2efD+JMog/vVeBP3BJcj/goIQ/87CQP2UfjD8A/G0/75pfP08LcD9rh4A/rnpEPyBmej/M"
        "GmA/A0FtPyURcj8Y53c/1USFPxxYjT+jtXw/oX2RP5X4bj/Uf4Q/fpCLPw80gT9n92w/YWdoPz9I"
        "dD+J0YI/XidmPyalej+B7Hs/PuyGP2W/gj98jIQ/EkNvP5CufD/uCIo/rR2TP5fEXz/TYJM/KjqR"
        "PzQAij+RYoM/rfZ3P6Spkj9gF5k/mQ+XP1XVkD8Qk4Q/KhFhP8/ifz8jZ4g/mARfP7wOhT+UgIU/"
        "y+iIP8SvYT93D28/x9N0P5YNYj+QQ5Y/AE5zP/g1hD9rYXk/uUSUP4/mkD9fG4g/FJdHP32qgD9O"
        "wIg/yNmMP3wucD9eUpc/bDJeP5wQbz/594s/vqCAP3GhmT+9aYI/TcpvP5pVdj8REWQ/mEpfP7sR"
        "iD9dcIc/A5KQP56ubD/enpU/k6R4PwUnlD+y63Q/8CttP38ygz/eM40/mQ+CP64CcT8vql0//x5c"
        "PzFvhj8Xq4w/R8t7Pwt/ZD/JLIs/zzhfP9a+bT/08oc/emVGPw7yhD8pHHE/FmaBP+IPfj9KloI/"
        "quKIP/OVbD9GMJI/REuJP77Mij8H6Y4/xRSKP+HNij+194A/f3lbP/iKfD/pTGw/65NbP+ZOgz/0"
        "cXE/E6NlP5dMZT+Nb4M/TJeEP27tkD/PpH8/51WNP/HykT/duI4/v3JDPye6jz/eWIQ/nWyFP3DA"
        "hD845oQ/qBaEP9PPdj9xUU8/NzZ9P6psaz97040/ipt4P4gRgT8GQGo/lu1yP2u0fz9y+Vk/SdmD"
        "P9hIfT9EpWE/8ZpCPyuRhj/BYXg/iW5yP1b0eT86QJc/oLl+P9YbgT9S7lk/ABaVP22+iz8iqI0/"
        "N5yAP7K7iz+Fv4Q/TNmHP5cafD+6RFo/WiuNPwx3Tj+N23k/pMN6P4NNZT8V2Yc/H996P/HQdD9r"
        "p4Y/scxzP2nHkT+mf4Q/adtzPxE6Tj+ChV4/VOmNP1y0fj+DwHg/mwiVPwgqXz/VAXE/2eZzP0+B"
        "hz91A28/6EtwP37oVj/vVYk/j1GKPwTOcz87F4I/hOheP+3rcz9Fo5E/w7yBP5mSnT8O2Ws/em2H"
        "P7z+ej8SPoc/vdB/PyKicT/9yWk/yj6nPx0Ffj+dX0w/VWVvP82tiD8lM3M/6WmRP6nUjD+iGXw/"
        "SelzP/BGZj+zFG4/nElaP5Fqjz9pXJQ/xtdfP7m/YT/huVI/SVNnP1B6MD/2wWI/u5mQP5omdz9N"
        "8Io/fntzP1uJlj/MjII/gzh2P8iroD+LsXc/l7xgP56Vgj9+AX8/IqaNP/tnaD/lTIo/SeqKPw=="
    ), dtype=np.float32).copy().reshape([256]), 'n0_crms_g2')
    i_2 = numpy_helper.from_array(np.array([9.999999974752427e-07], dtype=np.float32).reshape([1]), 'n0_crms_eps')
    i_3 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    ), dtype=np.float32).copy().reshape([1, 256]), 'n100_rarelu_res')
    i_4 = numpy_helper.from_array(np.array([0.17499999701976776], dtype=np.float32).reshape([1]), 'n200_rrid_slope')
    i_5 = numpy_helper.from_array(np.array([0.0], dtype=np.float32).reshape([1]), 'n200_rrid_zero')
    i_6 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([]), 'n300_cs_ax')
    i_7 = numpy_helper.from_array(np.array([0], dtype=np.int64).reshape([1]), 'n400_gsc_idx')
    i_8 = numpy_helper.from_array(np.array([0], dtype=np.int64).reshape([1]), 'n400_gsc_st')
    i_9 = numpy_helper.from_array(np.array([256], dtype=np.int64).reshape([1]), 'n400_gsc_en')
    i_10 = numpy_helper.from_array(np.array([1, 256], dtype=np.int64).reshape([2]), 'n400_gsc_exp')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('Mul', inputs=['model_input', 'model_input'], outputs=['n0_crms_sq']),
        helper.make_node('ReduceMean', inputs=['n0_crms_sq'], outputs=['n0_crms_ms'], axes=[-1], keepdims=1),
        helper.make_node('Add', inputs=['n0_crms_ms', 'n0_crms_eps'], outputs=['n0_crms_add']),
        helper.make_node('Sqrt', inputs=['n0_crms_add'], outputs=['n0_crms_rt']),
        helper.make_node('Div', inputs=['model_input', 'n0_crms_rt'], outputs=['n0_crms_n']),
        helper.make_node('Mul', inputs=['n0_crms_n', 'n0_crms_g'], outputs=['n0_crms_sc']),
        helper.make_node('Mul', inputs=['n0_crms_sc', 'n0_crms_g2'], outputs=['n0_crms_out']),
        helper.make_node('Add', inputs=['n0_crms_out', 'n100_rarelu_res'], outputs=['n100_rarelu_add']),
        helper.make_node('Relu', inputs=['n100_rarelu_add'], outputs=['n100_rarelu_out']),
        helper.make_node('Mul', inputs=['n100_rarelu_out', 'n200_rrid_slope'], outputs=['n200_rrid_scl']),
        helper.make_node('Greater', inputs=['n100_rarelu_out', 'n200_rrid_zero'], outputs=['n200_rrid_gt']),
        helper.make_node('Where', inputs=['n200_rrid_gt', 'n100_rarelu_out', 'n200_rrid_scl'], outputs=['n200_rrid_out']),
        helper.make_node('CumSum', inputs=['n200_rrid_out', 'n300_cs_ax'], outputs=['n300_cs_out']),
        helper.make_node('Gather', inputs=['n300_cs_out', 'n400_gsc_idx'], outputs=['n400_gsc_g'], axis=0),
        helper.make_node('Slice', inputs=['n400_gsc_g', 'n400_gsc_st', 'n400_gsc_en'], outputs=['n400_gsc_sl']),
        helper.make_node('Expand', inputs=['n400_gsc_sl', 'n400_gsc_exp'], outputs=['n400_gsc_out']),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000092", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "Qf47PmvcaLxKT/m7oNPhPdh2lL7D1k++rlwGP4R5SL8z15C9DrpLvG75rL46OXs/1fPlvU9LHb8w"
        "fqQ+rOPRvbexsb8jRBG+DNiTvj3c+j2dg8e+9NhGP8TQej9saxU/JysIP95xI7403ba+zasvPswL"
        "GT+Wdzu/SvwBvyO2qL5UeW6+dpfbPjYn6r7SNva+cOynvudLmr26u4A+NTBOP9dVB79n0+k89GYH"
        "vpppmD0hnqg+5L8pP/Y1Gb+OaSU/lxJFPv+VKD97p3m+D4G5vRdKsb6gbk6+Zs+4Pn5fUb/AadY9"
        "402UPdJIib8hZRA96bAsP7U+8D5ApAc8aaeGP27n8D7OsCC+LVXBvtf7i70yIfi+yhksPc+9rb8R"
        "OoI+9GUgv3SOwT2YoX6+AiUVPlbmED86ipe+RdzjvkFtwb6F9di+D2o+PvBX3j6+LUA8im3Xvs7A"
        "Ej02OAc/u8kTvpdUQT+8eCo+aKKvviYGRD842hG/rzLhPo8ilD4o4Iu9iHK7PpNvx7//wd6+WDmn"
        "PhqQ9b0/uxm/amc7v6KxED/+tYe9RQUuPvDRiT6mtHU/MF0KvrYulD4MAIe+M8QcO/+Wsz6b0y++"
        "u2WUP3fkEj1R1IA9F4PEPKuhFD8m00K+qw5vPtPhQL5xKoq+3Fxcvyrucr5HdPe9pe4LPuyMVz+0"
        "K3G/n69COld3477hJCC9q9XHvqeIkT8weSQ+LaVWP1ja2766E+G9dnm0Owz8LD4s/Z++rMRSP6tQ"
        "RT6wOaK+/z6QP6MXhD/Wwg++2Y5/PrWzn77yQQG/lCmiv7jofD5yDzW/g2qrPV+udj+Uocw9Pl9Z"
        "v33pib7AHz8+eHMvvoAqD78+vzC/C1vsvkOuUr/FGne/RgRzv0PxUr8Mdyw9+34BPowIIb/FcAi/"
        "15TcPtsNhb5Wdqw+/jvJPQ7Qpz5qr0o+fCw/P20vZL7s0Mc+k4F9P7a/gb2DPje/6WSePm1tZb1s"
        "91k9cF0Qv2Jojb4bHJU/XrkAP6VGDT+uUT8/bemivk39Or8Z7ui+OzW9Pbr5kz7jDnW+H7LQPgfX"
        "Kz/cwCM9ryGVvo5qFj8AILm+SGsCP5qRMj01N/M9Kcv6vl4S/D4nbca+phcjv9uKv70F6Ya9I9lP"
        "vhHbTD7urmM+9HZRPQl/ST5T2A6/UVmSPkHy5j7yQeS+fyldvgB4lD51EW8/3rRfvqADfb6jGDK/"
        "rZirPmJ3Fr29lJw+UxUqPkulRz6p7tA90DXUvplhMr3zIZ4+esQdPw+40748PfK+Dby1v++Woz1u"
        "q/m+DWwkPOHxqT5wB/28vEOgvkgDXz9DVts+id32Poks7T7GV8s+LjIUvmmjXD8ekRw/okSGvg=="
    ), dtype=np.float32).copy().reshape([1, 256])


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
