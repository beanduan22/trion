#!/usr/bin/env python3
"""
Bug ID     : bug_000041
Source     : Trion campaign v4 (minimized via delta-debug)
Compiler   : openvino, tvm, xla
Patterns   : LayerNormalization, Dropout, TopK, Tile, Abs, Mul, Gather, Slice...
Root cause : openvino rel_L2=1.000
Minimal ops: LayerNormalization -> Dropout -> TopK -> Tile -> Abs -> Mul -> Gather -> Slice -> Expand -> LayerNormalization (10 ops, down from 11)
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
BACKENDS = ['openvino', 'tvm', 'xla']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 128]
OUTPUT_NAME = 'n400_lnr_ln'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8="
    ), dtype=np.float32).copy().reshape([128]), 'n0_lnd_sc')
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
    ), dtype=np.float32).copy().reshape([128]), 'n0_lnd_b')
    i_2 = numpy_helper.from_array(np.array([0.0], dtype=np.float32).reshape([]), 'n0_lnd_ratio')
    i_3 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([1]), 'n100_tk_k')
    i_4 = numpy_helper.from_array(np.array([1, 128], dtype=np.int64).reshape([2]), 'n100_tk_rep')
    i_5 = numpy_helper.from_array(np.array([0], dtype=np.int64).reshape([1]), 'n300_gsc_idx')
    i_6 = numpy_helper.from_array(np.array([0], dtype=np.int64).reshape([1]), 'n300_gsc_st')
    i_7 = numpy_helper.from_array(np.array([128], dtype=np.int64).reshape([1]), 'n300_gsc_en')
    i_8 = numpy_helper.from_array(np.array([1, 128], dtype=np.int64).reshape([2]), 'n300_gsc_exp')
    i_9 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8="
    ), dtype=np.float32).copy().reshape([128]), 'n400_lnr_sc')
    i_10 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    ), dtype=np.float32).copy().reshape([128]), 'n400_lnr_b')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10]


def _nodes():
    return [
        helper.make_node('LayerNormalization', inputs=['model_input', 'n0_lnd_sc', 'n0_lnd_b'], outputs=['n0_lnd_ln'], axis=-1, epsilon=9.999999747378752e-06),
        helper.make_node('Dropout', inputs=['n0_lnd_ln', 'n0_lnd_ratio'], outputs=['n0_lnd_out']),
        helper.make_node('TopK', inputs=['n0_lnd_out', 'n100_tk_k'], outputs=['n100_tk_vals', 'n100_tk_idx'], axis=-1, largest=1, sorted=1),
        helper.make_node('Tile', inputs=['n100_tk_vals', 'n100_tk_rep'], outputs=['n100_tk_out']),
        helper.make_node('Abs', inputs=['n100_tk_out'], outputs=['n200_abs__a']),
        helper.make_node('Mul', inputs=['n200_abs__a', 'n100_tk_out'], outputs=['n200_abs__out']),
        helper.make_node('Gather', inputs=['n200_abs__out', 'n300_gsc_idx'], outputs=['n300_gsc_g'], axis=0),
        helper.make_node('Slice', inputs=['n300_gsc_g', 'n300_gsc_st', 'n300_gsc_en'], outputs=['n300_gsc_sl']),
        helper.make_node('Expand', inputs=['n300_gsc_sl', 'n300_gsc_exp'], outputs=['n300_gsc_out']),
        helper.make_node('LayerNormalization', inputs=['n300_gsc_out', 'n400_lnr_sc', 'n400_lnr_b'], outputs=['n400_lnr_ln'], axis=-1, epsilon=9.999999747378752e-06),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000041_min", inputs_info, outputs_info, initializer=_inits())
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = IR_VERSION
    return m.SerializeToString()


def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "uDjjP1nkPr/kOIS/SYksQCc72b9u8ZPANGm6Pgd3pL8yVue//JnlvS8g7r60vvw/R9diP8Vhs74d"
        "rQPAfBInP2DFg7/xuv6+Q3X6Pju6vb/kLPm+xKuzPytErz8ldxXAZqvQv3OspD/YJzG/x1y/Ps8S"
        "xj/JKjk/MPY1QCzZbED9XgfAEEqiwBZ1Sb/VD7e/uXQ+wJfJi8BstCHAqit9PyrzFcAmoH9A/1ZO"
        "v/isR8AaSMC/SO3xv+5mm78sqynA4/L3Pxq5or63jqlA409nv05x0j+9BaY/mtxiQKBVPD/OOFlA"
        "qaKLQO7AOECw/Yy+2s9rPlnNRj/N7bI+Yu28P51vBL+Pb7E/p/Zxv6XPkT4bDRQ/Ig+yvyx0EkB6"
        "kBu/yNRLP6TaxD8ty+E/cAI9wPERrj/BYdc+NqzEPe7p/T/qp/W+wckgwEfRwECybIlAH1MgvabF"
        "CsA8xNy/ehcmP3ZRjD8hjgrAUSc+P6p4wj5Tm+2/N8eJQIVDcj8/1wc/sAeEQOGdpT9iF+49W7+R"
        "P0d1RL8Odti/3ZRvPzZmBEBgeorA04jiP7pX4r5jPmLA9GFXQAjIlT/xqlRAkwOMwOij8D6gyQ5A"
        "8SXEv2ZZL8AK6SPAEPSzv3p+Kb8tgmK/C8ImP8LV7z6nX2W/pgWivydtCT9XiKvAc+NZQL2bxr8="
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
