#!/usr/bin/env python3
"""
Cross-compiler differential testing harness — extended patterns.
Tests 18 new ONNX patterns (Groups A–D) against all 5 compilers.

Oracle : ORT with ORT_DISABLE_ALL (no optimization)
Tolerance : 0.05  (tighter than existing harness's 0.1)

Compilers tested:
  1. ORT_opt  (ORT_ENABLE_ALL)
  2. OpenVINO (CPU plugin)
  3. onnx2torch (torch eager)
  4. torch.compile (Inductor)
  5. GPU(torch)  — uses GPU if available, else skips
"""

import numpy as np
import onnx
import warnings
import sys
import os
import traceback
from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
import onnxruntime as ort

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(42)
TOL = 0.05

# ── Helpers ──────────────────────────────────────────────────────────────────

def mk(nodes, in_infos, out_name, out_shape, out_dtype, inits, opset=13):
    g = oh.make_graph(nodes, "t", in_infos,
        [oh.make_tensor_value_info(out_name, out_dtype, out_shape)],
        initializer=inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 8
    return m.SerializeToString()

def mkf(nodes, in_infos, out_name, out_shape, inits, opset=13):
    """Float32 output model (shortcut)."""
    return mk(nodes, in_infos, out_name, out_shape, TP.FLOAT, inits, opset)


# ── Compiler runners ─────────────────────────────────────────────────────────

def run_ort_ref(mb, feed):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"]).run(None, feed)[0]

def run_ort_opt(mb, feed):
    return ort.InferenceSession(mb,
                                providers=["CPUExecutionProvider"]).run(None, feed)[0]

def run_openvino(mb, feed):
    import openvino as ov
    core = ov.Core()
    m = core.read_model(mb, b'')
    c = core.compile_model(m, "CPU")
    return c(feed)[c.output(0)]

def run_torch_eager(mb, feed):
    from onnx2torch import convert
    import torch
    m = onnx.load_from_string(mb)
    tm = convert(m).eval()
    k = list(feed.keys())[0]
    with torch.no_grad():
        out = tm(torch.from_numpy(feed[k]))
        return out.numpy() if out.dtype != torch.float16 else out.float().numpy()

def run_torch_compile(mb, feed):
    from onnx2torch import convert
    import torch
    m = onnx.load_from_string(mb)
    tm = convert(m).eval()
    k = list(feed.keys())[0]
    cm = torch.compile(tm, fullgraph=False)
    with torch.no_grad():
        out = cm(torch.from_numpy(feed[k]))
        return out.numpy() if out.dtype != torch.float16 else out.float().numpy()

def run_gpu_torch(mb, feed):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    from onnx2torch import convert
    m = onnx.load_from_string(mb)
    tm = convert(m).eval().cuda()
    k = list(feed.keys())[0]
    x_gpu = torch.from_numpy(feed[k]).cuda()
    with torch.no_grad():
        out = tm(x_gpu).cpu()
        return out.numpy() if out.dtype != torch.float16 else out.float().numpy()


# ── Test registry ─────────────────────────────────────────────────────────────

TESTS = []

def t(name, category="general"):
    def d(fn):
        TESTS.append((name, category, fn))
        return fn
    return d


# ============================================================
# GROUP A — ORT-specific patterns, check cross-compiler
# ============================================================

@t("layernorm_fusion", "A_ort")
def _():
    """LayerNorm manual pattern: ORT fuses it, others compute differently?"""
    batch, seq, hidden = 1, 8, 64
    x = np.random.randn(batch, seq, hidden).astype(np.float32)
    scale = np.ones(hidden, dtype=np.float32)
    bias  = np.zeros(hidden, dtype=np.float32)
    eps   = np.array(1e-5, dtype=np.float32)
    two   = np.array(2.0, dtype=np.float32)

    nodes = [
        oh.make_node("ReduceMean", ["X"],          ["mean"],    axes=[-1], keepdims=1),
        oh.make_node("Sub",        ["X", "mean"],  ["diff"]),
        oh.make_node("Pow",        ["diff", "two"],["sq"]),
        oh.make_node("ReduceMean", ["sq"],         ["var"],     axes=[-1], keepdims=1),
        oh.make_node("Add",        ["var", "eps"], ["var_eps"]),
        oh.make_node("Sqrt",       ["var_eps"],    ["std"]),
        oh.make_node("Div",        ["diff", "std"],["norm"]),
        oh.make_node("Mul",        ["norm", "sc"], ["scaled"]),
        oh.make_node("Add",        ["scaled", "bi"],["Y"]),
    ]
    inits = [
        onh.from_array(scale, "sc"),
        onh.from_array(bias, "bi"),
        onh.from_array(eps, "eps"),
        onh.from_array(two, "two"),
    ]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [batch, seq, hidden])],
               "Y", [batch, seq, hidden], inits), {"X": x}


@t("skip_layernorm_fp16", "A_ort")
def _():
    """FP16 LayerNorm: Cast→fp16, LayerNorm op, Cast back."""
    batch, seq, hidden = 1, 8, 64
    x = np.random.randn(batch, seq, hidden).astype(np.float32) * 2

    nodes = [
        oh.make_node("Cast", ["X"], ["xh"], to=TP.FLOAT16),
        oh.make_node("ReduceMean", ["xh"],         ["mean_h"], axes=[-1], keepdims=1),
        oh.make_node("Sub",        ["xh","mean_h"],["diff_h"]),
        oh.make_node("Mul",        ["diff_h","diff_h"],["sq_h"]),
        oh.make_node("ReduceMean", ["sq_h"],       ["var_h"],  axes=[-1], keepdims=1),
        oh.make_node("Add",        ["var_h","eps_h"],["ve_h"]),
        oh.make_node("Sqrt",       ["ve_h"],       ["std_h"]),
        oh.make_node("Div",        ["diff_h","std_h"],["norm_h"]),
        oh.make_node("Cast",       ["norm_h"],     ["Y"],       to=TP.FLOAT),
    ]
    inits = [
        onh.from_array(np.array(1e-5, dtype=np.float16), "eps_h"),
    ]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [batch, seq, hidden])],
               "Y", [batch, seq, hidden], inits), {"X": x}


@t("cast_fp16_int32_fp16", "A_ort")
def _():
    """fp16 → int32 → fp16 roundtrip (fp16 version of cast_fp32_int32_fp32)."""
    x = np.array([[0.1, 0.6, 1.4, -0.7, 2.9, -1.1, 0.5, 1.5]],
                 dtype=np.float16)
    x_f32 = x.astype(np.float32)
    nodes = [
        oh.make_node("Cast", ["X"], ["xf16"], to=TP.FLOAT16),
        oh.make_node("Cast", ["xf16"], ["xi32"], to=TP.INT32),
        oh.make_node("Cast", ["xi32"], ["Y"],    to=TP.FLOAT),
    ]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [1, 8])],
               "Y", [1, 8], []), {"X": x_f32}


@t("convtranspose_autopad_same", "A_ort")
def _():
    """ConvTranspose with auto_pad=SAME_UPPER."""
    np.random.seed(7)
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    w = np.random.randn(1, 1, 3, 3).astype(np.float32)
    node = oh.make_node("ConvTranspose", ["X", "W"], ["Y"],
                        auto_pad="SAME_UPPER", strides=[2, 2])
    return mkf([node],
               [oh.make_tensor_value_info("X", TP.FLOAT, [1, 1, 4, 4])],
               "Y", [1, 1, 8, 8],
               [onh.from_array(w, "W")], opset=11), {"X": x}


@t("convtranspose_dilated", "A_ort")
def _():
    """ConvTranspose with dilation=2."""
    np.random.seed(13)
    x = np.random.randn(1, 2, 5, 5).astype(np.float32)
    w = np.random.randn(2, 1, 3, 3).astype(np.float32)
    node = oh.make_node("ConvTranspose", ["X", "W"], ["Y"],
                        dilations=[2, 2], strides=[2, 2],
                        output_padding=[1, 1], pads=[0, 0, 0, 0])
    return mk([node],
              [oh.make_tensor_value_info("X", TP.FLOAT, [1, 2, 5, 5])],
              "Y", None, TP.FLOAT,
              [onh.from_array(w, "W")], opset=11), {"X": x}


@t("instance_norm_fp16", "A_ort")
def _():
    """InstanceNorm on fp16 input [2, 4, 4, 4]."""
    np.random.seed(41)
    x_fp32 = (np.random.randn(2, 4, 4, 4) * 20).astype(np.float32)
    # Build fp32 model but with fp16 intermediate via cast
    nodes = [
        oh.make_node("Cast", ["X"], ["Xh"], to=TP.FLOAT16),
        oh.make_node("Cast", ["sc"], ["sch"], to=TP.FLOAT16),
        oh.make_node("Cast", ["bi"], ["bih"], to=TP.FLOAT16),
        oh.make_node("InstanceNormalization", ["Xh", "sch", "bih"], ["Yh"], epsilon=1e-5),
        oh.make_node("Cast", ["Yh"], ["Y"], to=TP.FLOAT),
    ]
    inits = [
        onh.from_array(np.ones(4, dtype=np.float32), "sc"),
        onh.from_array(np.zeros(4, dtype=np.float32), "bi"),
    ]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [2, 4, 4, 4])],
               "Y", [2, 4, 4, 4], inits, opset=13), {"X": x_fp32}


@t("scatternd_basic", "A_ort")
def _():
    """ScatterND basic update on [4, 4] tensor."""
    data    = np.arange(16, dtype=np.float32).reshape(4, 4)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.ones((2, 4), dtype=np.float32) * 99.0

    node = oh.make_node("ScatterND", ["D", "I", "U"], ["Y"], reduction="none")
    g = oh.make_graph([node], "t",
        [oh.make_tensor_value_info("D", TP.FLOAT, [4, 4]),
         oh.make_tensor_value_info("I", TP.INT64, [2, 1]),
         oh.make_tensor_value_info("U", TP.FLOAT, [2, 4])],
        [oh.make_tensor_value_info("Y", TP.FLOAT, [4, 4])],
        initializer=[])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 16)])
    m.ir_version = 8
    return m.SerializeToString(), {"D": data, "I": indices, "U": updates}


@t("gridsample_bilinear_zeros", "A_ort")
def _():
    """GridSample bilinear + zeros padding."""
    feat = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    grid = np.array([[[[-0.5, -0.5], [0.5, -0.5],
                       [-0.5,  0.5], [0.5,  0.5]]]],
                    dtype=np.float32)
    node = oh.make_node("GridSample", ["X", "grid"], ["Y"],
                        mode="bilinear", padding_mode="zeros", align_corners=0)
    g = oh.make_graph([node], "t",
        [oh.make_tensor_value_info("X",    TP.FLOAT, [1, 1, 4, 4]),
         oh.make_tensor_value_info("grid", TP.FLOAT, [1, 1, 4, 2])],
        [oh.make_tensor_value_info("Y",    TP.FLOAT, [1, 1, 1, 4])],
        initializer=[])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 16)])
    m.ir_version = 8
    return m.SerializeToString(), {"X": feat, "grid": grid}


# ============================================================
# GROUP B — OpenVINO-specific patterns, cross-check ORT/torch
# ============================================================

@t("reducesum_mul_add", "B_openvino")
def _():
    """Mul([1,32,4] × [32,4]) + ReduceSum(axis=-1) + Add([32])."""
    np.random.seed(0)
    x_np      = np.random.randn(1, 32, 4).astype(np.float32)
    weight_np = np.random.rand(32, 4).astype(np.float32)
    bias_np   = np.random.rand(32).astype(np.float32)

    nodes = [
        oh.make_node("Mul",       ["X", "W"],   ["mul_out"]),
        oh.make_node("ReduceSum", ["mul_out", "ax"], ["rs_out"], keepdims=0),
        oh.make_node("Add",       ["rs_out", "B"],  ["Y"]),
    ]
    inits = [
        onh.from_array(weight_np, "W"),
        onh.from_array(bias_np,   "B"),
        onh.from_array(np.array([-1], dtype=np.int64), "ax"),
    ]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [1, 32, 4])],
               "Y", [1, 32], inits, opset=13), {"X": x_np}


@t("matmul_large_dim", "B_openvino")
def _():
    """MatMul [1, 1, 2049] × [1, 2049, 1] — triggers OV GPU tile bug."""
    size = 2049
    A = np.ones((1, 1, size), dtype=np.float32)
    B = np.ones((1, size, 1), dtype=np.float32)
    node = oh.make_node("MatMul", ["A", "B"], ["Y"])
    g = oh.make_graph([node], "t",
        [oh.make_tensor_value_info("A", TP.FLOAT, [1, 1, size]),
         oh.make_tensor_value_info("B", TP.FLOAT, [1, size, 1])],
        [oh.make_tensor_value_info("Y", TP.FLOAT, [1, 1, 1])],
        initializer=[])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 8
    return m.SerializeToString(), {"A": A, "B": B}


@t("softsign_activation", "B_openvino")
def _():
    """Softsign on linspace(-5,5,64): OV returns all-ones historically."""
    x = np.linspace(-5, 5, 64).reshape(1, 64).astype(np.float32)
    return mkf([oh.make_node("Softsign", ["X"], ["Y"])],
               [oh.make_tensor_value_info("X", TP.FLOAT, [1, 64])],
               "Y", [1, 64], [], opset=9), {"X": x}


# ============================================================
# GROUP C — Optimization divergence patterns
# ============================================================

@t("sqrt_div_precision", "C_optdiv")
def _():
    """sqrt(x)/y where x=0.0001 — rsqrt approximation precision."""
    x = np.array([[0.0001]], dtype=np.float32)
    y = np.array([[1.0]], dtype=np.float32)
    nodes = [
        oh.make_node("Sqrt", ["X"], ["sqx"]),
        oh.make_node("Div",  ["sqx", "Y"], ["Z"]),
    ]
    g = oh.make_graph(nodes, "t",
        [oh.make_tensor_value_info("X", TP.FLOAT, [1, 1]),
         oh.make_tensor_value_info("Y", TP.FLOAT, [1, 1])],
        [oh.make_tensor_value_info("Z", TP.FLOAT, [1, 1])],
        initializer=[])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 8
    return m.SerializeToString(), {"X": x, "Y": y}


@t("const_fold_add_chain", "C_optdiv")
def _():
    """Add(Add(x, c1), c2) — constant fold may accumulate differently."""
    x  = np.random.randn(1, 8).astype(np.float32)
    c1 = np.array([0.1] * 8, dtype=np.float32)
    c2 = np.array([0.2] * 8, dtype=np.float32)
    nodes = [
        oh.make_node("Add", ["X", "c1"], ["t1"]),
        oh.make_node("Add", ["t1", "c2"], ["Y"]),
    ]
    inits = [onh.from_array(c1, "c1"), onh.from_array(c2, "c2")]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [1, 8])],
               "Y", [1, 8], inits), {"X": x}


@t("fp16_matmul_add", "C_optdiv")
def _():
    """fp16 MatMul + Add — OpenVINO CPU fp16 MatMul accumulation differs from ORT.
    Size 64 gives max_diff ~0.078 (>0.05 threshold)."""
    np.random.seed(42)
    size = 64
    W = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size).astype(np.float16)
    x = np.random.randn(1, size).astype(np.float16)

    W_init = onh.from_array(W, "W")
    B_init = onh.from_array(B, "B")

    nodes = [
        oh.make_node("MatMul", ["x", "W"],       ["mm_out"]),
        oh.make_node("Add",    ["mm_out", "B"],   ["Y"]),
    ]
    g = oh.make_graph(nodes, "t",
        [oh.make_tensor_value_info("x", TP.FLOAT16, [1, size])],
        [oh.make_tensor_value_info("Y", TP.FLOAT16, [1, size])],
        initializer=[W_init, B_init])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    return m.SerializeToString(), {"x": x}


@t("layernorm_residual", "C_optdiv")
def _():
    """LayerNorm + Add residual [1, 32, 64]."""
    batch, seq, hidden = 1, 32, 64
    x = np.random.randn(batch, seq, hidden).astype(np.float32)
    two = np.array(2.0, dtype=np.float32)
    eps = np.array(1e-5, dtype=np.float32)
    scale = np.ones(hidden, dtype=np.float32)
    bias  = np.zeros(hidden, dtype=np.float32)

    nodes = [
        oh.make_node("ReduceMean", ["X"],          ["mean"],    axes=[-1], keepdims=1),
        oh.make_node("Sub",        ["X", "mean"],  ["diff"]),
        oh.make_node("Pow",        ["diff", "two"],["sq"]),
        oh.make_node("ReduceMean", ["sq"],         ["var"],     axes=[-1], keepdims=1),
        oh.make_node("Add",        ["var", "eps"], ["var_eps"]),
        oh.make_node("Sqrt",       ["var_eps"],    ["std"]),
        oh.make_node("Div",        ["diff", "std"],["norm"]),
        oh.make_node("Mul",        ["norm", "sc"], ["scaled"]),
        oh.make_node("Add",        ["scaled", "bi"],["ln_out"]),
        oh.make_node("Add",        ["ln_out", "X"],["Y"]),   # residual
    ]
    inits = [
        onh.from_array(scale, "sc"),
        onh.from_array(bias, "bi"),
        onh.from_array(eps, "eps"),
        onh.from_array(two, "two"),
    ]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [batch, seq, hidden])],
               "Y", [batch, seq, hidden], inits), {"X": x}


# ============================================================
# GROUP D — Inductor patterns as ONNX
# ============================================================

@t("softshrink_pattern", "D_inductor")
def _():
    """Manual softshrink: x where |x|>0.5, else 0, via Where."""
    x = np.linspace(-2, 2, 32).reshape(1, 32).astype(np.float32)
    lambd = np.float32(0.5)

    nodes = [
        # shrink positive side: x - 0.5
        oh.make_node("Sub",     ["X",  "lam"],   ["x_sub"]),
        # shrink negative side: x + 0.5
        oh.make_node("Add",     ["X",  "lam"],   ["x_add"]),
        # |x| > 0.5
        oh.make_node("Abs",     ["X"],           ["absx"]),
        oh.make_node("Greater", ["absx", "lam"], ["mask"]),
        # sign(x): x > 0 → +1; x < 0 → -1
        oh.make_node("Greater", ["X", "zero"],   ["pos_mask"]),
        oh.make_node("Where",   ["pos_mask", "x_sub", "x_add"], ["shrunk"]),
        oh.make_node("Where",   ["mask", "shrunk", "zero_f"],   ["Y"]),
    ]
    inits = [
        onh.from_array(np.array(lambd, dtype=np.float32),       "lam"),
        onh.from_array(np.array(0.0,   dtype=np.float32),       "zero"),
        onh.from_array(np.zeros(32,    dtype=np.float32),       "zero_f"),
    ]
    return mkf(nodes,
               [oh.make_tensor_value_info("X", TP.FLOAT, [1, 32])],
               "Y", [1, 32], inits), {"X": x}


@t("batchnorm_eval", "D_inductor")
def _():
    """BatchNormalization in inference mode [2, 8, 4, 4]."""
    np.random.seed(99)
    x = np.random.randn(2, 8, 4, 4).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, 8).astype(np.float32)
    bias  = np.random.uniform(-1, 1, 8).astype(np.float32)
    mean  = np.random.uniform(-0.5, 0.5, 8).astype(np.float32)
    var   = np.random.uniform(0.1, 1.0, 8).astype(np.float32)

    node = oh.make_node("BatchNormalization", ["X", "sc", "bi", "mn", "vr"], ["Y"],
                        epsilon=1e-5, training_mode=0)
    inits = [
        onh.from_array(scale, "sc"),
        onh.from_array(bias,  "bi"),
        onh.from_array(mean,  "mn"),
        onh.from_array(var,   "vr"),
    ]
    return mkf([node],
               [oh.make_tensor_value_info("X", TP.FLOAT, [2, 8, 4, 4])],
               "Y", [2, 8, 4, 4], inits, opset=15), {"X": x}


@t("pixel_shuffle", "D_inductor")
def _():
    """DepthToSpace [1, 16, 4, 4] → [1, 4, 8, 8] (pixel shuffle)."""
    x = np.random.randn(1, 16, 4, 4).astype(np.float32)
    node = oh.make_node("DepthToSpace", ["X"], ["Y"],
                        blocksize=2, mode="DCR")
    return mkf([node],
               [oh.make_tensor_value_info("X", TP.FLOAT, [1, 16, 4, 4])],
               "Y", [1, 4, 8, 8], [], opset=13), {"X": x}


# ── Run all tests ─────────────────────────────────────────────────────────────

def diff(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

COMPILERS = [
    ("ORT_opt",     run_ort_opt),
    ("OpenVINO",    run_openvino),
    ("onnx2torch",  run_torch_eager),
    ("torch.compile", run_torch_compile),
    ("GPU(torch)",  run_gpu_torch),
]

if __name__ == "__main__":
    hdr = f"{'Pattern':<35}"
    for cname, _ in COMPILERS:
        hdr += f" {cname:<15}"
    hdr += " Status"
    print(hdr)
    print("=" * (len(hdr) + 10))

    all_bugs = []
    summary  = {c: {"bug": 0, "ok": 0, "err": 0} for c, _ in COMPILERS}
    bug_details = []  # (name, category, cname, d, ref, out)

    for name, cat, builder in TESTS:
        try:
            mb, feed = builder()
        except Exception as e:
            print(f"{name:<35} BUILD ERROR: {e}")
            continue

        try:
            ref = run_ort_ref(mb, feed).astype(np.float64)
        except Exception as e:
            print(f"{name:<35} REF ERROR: {e}")
            continue

        row = f"{name:<35}"
        row_bugs = []

        for cname, runner in COMPILERS:
            try:
                out = runner(mb, feed).astype(np.float64)
                d   = diff(out, ref)
                if d > TOL:
                    row += f" {d:<15.4f}"
                    row_bugs.append((cname, d))
                    all_bugs.append((name, cname, d))
                    summary[cname]["bug"] += 1
                    bug_details.append((name, cat, cname, d, ref.copy(), out.copy()))
                else:
                    row += f" {d:<15.2e}"
                    summary[cname]["ok"] += 1
            except Exception as e:
                short = str(e)[:30]
                row  += f" {'ERR:'+short:<15}"[:17]
                row  += " " * max(0, 16 - len("ERR:"+short))
                # fix formatting
                row = row[:35 + (list(COMPILERS).index((cname, runner)) + 1) * 16]
                summary[cname]["err"] += 1

        status = f"*** {len(row_bugs)} BUG(S) ***" if row_bugs else "OK"
        print(row + f" {status}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"TOLERANCE : {TOL}")
    print(f"PATTERNS  : {len(TESTS)}")
    print()

    print(f"{'Compiler':<20} {'Bugs':>5} {'OK':>5} {'Errors':>6}")
    print("-" * 40)
    total_bugs = 0
    for cname, _ in COMPILERS:
        s = summary[cname]
        print(f"{cname:<20} {s['bug']:>5} {s['ok']:>5} {s['err']:>6}")
        total_bugs += s["bug"]
    print(f"{'TOTAL':<20} {total_bugs:>5}")

    print(f"\n{'='*110}")
    print(f"ALL BUGS (tolerance > {TOL}):")
    print(f"{'Pattern':<35} {'Category':<15} {'Compiler':<20} {'max_diff':>10}")
    print("-" * 85)
    for name, cname, d in sorted(all_bugs, key=lambda x: -x[2]):
        # find category
        cat = next((c for n, c, fn in TESTS if n == name), "?")
        print(f"{name:<35} {cat:<15} {cname:<20} {d:>10.4f}")
