#!/usr/bin/env python3
"""
Bug ID     : bug_000223
Source     : Trion campaign v4 (minimized via delta-debug)
Compiler   : onnxruntime
Patterns   : Gather, Reshape, TopK, Tile, Abs, ReduceSum, Add, Div...
Root cause : onnxruntime rel_L2=1.000
Minimal ops: Gather -> Reshape -> TopK -> Tile -> Abs -> ReduceSum -> Add -> Div -> Add -> Mul -> Add -> LayerNormalization (12 ops, down from 14)
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
BACKENDS = ['onnxruntime']
INPUT_NAME = 'model_input'
INPUT_SHAPE = [1, 64]
OUTPUT_NAME = 'n400_lnt_ln'
OPSET = 17
IR_VERSION = 8


def _inits():
    i_0 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([64, 128]), 'n0_gr_emb')
    i_1 = numpy_helper.from_array(np.array([3], dtype=np.int64).reshape([1]), 'n0_gr_idx')
    i_2 = numpy_helper.from_array(np.array([1, 128], dtype=np.int64).reshape([2]), 'n0_gr_oshape')
    i_3 = numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([1]), 'n100_tk_k')
    i_4 = numpy_helper.from_array(np.array([1, 128], dtype=np.int64).reshape([2]), 'n100_tk_rep')
    i_5 = numpy_helper.from_array(np.array([1.0000000116860974e-07], dtype=np.float32).reshape([1]), 'n200_rbn_eps')
    i_6 = numpy_helper.from_array(np.array([-1], dtype=np.int64).reshape([1]), 'n200_rbn_axes')
    i_7 = numpy_helper.from_array(np.array([0.10000000149011612], dtype=np.float32).reshape([1]), 'n300_ama_a')
    i_8 = numpy_helper.from_array(np.array([1.5], dtype=np.float32).reshape([1]), 'n300_ama_b')
    i_9 = numpy_helper.from_array(np.array([0.20000000298023224], dtype=np.float32).reshape([1]), 'n300_ama_c')
    i_10 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
        "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
        "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
        "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8="
    ), dtype=np.float32).copy().reshape([128]), 'n400_lnt_sc')
    i_11 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    ), dtype=np.float32).copy().reshape([128]), 'n400_lnt_b')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10, i_11]


def _nodes():
    return [
        helper.make_node('Gather', inputs=['n0_gr_emb', 'n0_gr_idx'], outputs=['n0_gr_gath'], axis=0),
        helper.make_node('Reshape', inputs=['n0_gr_gath', 'n0_gr_oshape'], outputs=['n0_gr_out']),
        helper.make_node('TopK', inputs=['n0_gr_out', 'n100_tk_k'], outputs=['n100_tk_vals', 'n100_tk_idx'], axis=-1, largest=1, sorted=1),
        helper.make_node('Tile', inputs=['n100_tk_vals', 'n100_tk_rep'], outputs=['n100_tk_out']),
        helper.make_node('Abs', inputs=['n100_tk_out'], outputs=['n200_rbn_abs']),
        helper.make_node('ReduceSum', inputs=['n200_rbn_abs', 'n200_rbn_axes'], outputs=['n200_rbn_rs'], keepdims=1),
        helper.make_node('Add', inputs=['n200_rbn_rs', 'n200_rbn_eps'], outputs=['n200_rbn_add']),
        helper.make_node('Div', inputs=['n200_rbn_abs', 'n200_rbn_add'], outputs=['n200_rbn_out']),
        helper.make_node('Add', inputs=['n200_rbn_out', 'n300_ama_a'], outputs=['n300_ama_x1']),
        helper.make_node('Mul', inputs=['n300_ama_x1', 'n300_ama_b'], outputs=['n300_ama_x2']),
        helper.make_node('Add', inputs=['n300_ama_x2', 'n300_ama_c'], outputs=['n300_ama_out']),
        helper.make_node('LayerNormalization', inputs=['n300_ama_out', 'n400_lnt_sc', 'n400_lnt_b'], outputs=['n400_lnt_ln'], axis=-1, epsilon=9.999999747378752e-06),
    ]


def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None)]
    g = helper.make_graph(_nodes(), "bug_000223_min", inputs_info, outputs_info, initializer=_inits())
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = IR_VERSION
    return m.SerializeToString()


def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "7gWnv5M2hr9Zi78+SezdPsmDCL3asLW9NehvPh7y/L/rY6k/n6APP41AjD/KSFq/HkYvPxNHUr/j"
        "pyC/FFcjvjThl7/o/3C/NKbRPsuY/L+Bzhu+W6xovxrT774oz7E/GD0jvrMgEsAVkiI+R1QGwD6I"
        "1D7DMMC+bvJQPkBApD6BvME+rj2Hv/PVBb8drBm/8J9VvymiXr8LBh4+tNhMPwLBwL6Mo0C/7RyH"
        "v3oDQ8CuqtI/JTOdvpm8gz/NE6G/X2qiPuf+9T1CwrK/Su9APr5jCb2dlZI+sZS2PoIvdD0fzJe/"
        "zwUfv8zc9z4n2YE/N4+Gv2yZKj+ZwWc/TvkwPg=="
    ), dtype=np.float32).copy().reshape([1, 64])


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
