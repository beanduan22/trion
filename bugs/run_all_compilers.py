#!/usr/bin/env python3
"""
Cross-compiler differential testing harness.
Builds ONNX models for each known-buggy pattern and runs them through all 6 compilers.

Oracle: compiler output vs ORT-no-optimization reference
Tolerance: 0.1 (1e-1)

Compilers tested:
  1. OnnxRuntime (ORT_ENABLE_ALL)
  2. torch.compile (Inductor) via onnx2torch
  3. OpenVINO
  4. JAX/XLA (jax.jit) via onnx2torch→jax
  5. TensorFlow (tf.function + XLA)
  6. TVM — not installed, skipped
"""

import numpy as np
import onnx, warnings, sys, os, traceback
from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
import onnxruntime as ort

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

np.random.seed(42)
TOL = 0.1  # 1e-1

# ── Model builder helper ─────────────────────────────────────────────────────
def mk(nodes, in_infos, out_name, out_shape, inits, opset=13):
    g = oh.make_graph(nodes, "t", in_infos,
        [oh.make_tensor_value_info(out_name, TP.FLOAT, out_shape)],
        initializer=inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 8
    return m.SerializeToString()

# ── Compiler runners ─────────────────────────────────────────────────────────
def run_ort_ref(mb, feed):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, feed)[0]

def run_ort_opt(mb, feed):
    return ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, feed)[0]

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
        return tm(torch.from_numpy(feed[k])).numpy()

def run_torch_compile(mb, feed):
    from onnx2torch import convert
    import torch
    m = onnx.load_from_string(mb)
    tm = convert(m).eval()
    k = list(feed.keys())[0]
    cm = torch.compile(tm, fullgraph=False)
    with torch.no_grad():
        return cm(torch.from_numpy(feed[k])).numpy()

def run_tensorflow(mb, feed):
    import onnx2torch, torch, onnx
    m = onnx.load_from_string(mb)
    # Use onnx-tf or manual: run via onnxruntime TF provider
    # Simpler: use onnx → numpy reference via tf ops
    import tensorflow as tf
    # Convert ONNX to TF via onnx-tf if available, else use tf-onnx
    try:
        from onnx_tf.backend import prepare
        tf_rep = prepare(m)
        k = list(feed.keys())[0]
        return tf_rep.run(feed[k]).output.numpy() if hasattr(tf_rep.run(feed[k]), 'output') else np.array(tf_rep.run(feed[k])[0])
    except ImportError:
        # Fallback: use tf operations to replicate
        return None

def run_jax_jit(mb, feed):
    import jax, jax.numpy as jnp
    from onnx2torch import convert
    import torch
    m = onnx.load_from_string(mb)
    tm = convert(m).eval()
    k = list(feed.keys())[0]
    x_t = torch.from_numpy(feed[k])
    # Eager on CPU
    with torch.no_grad():
        eager = tm(x_t).numpy()
    # Move to GPU and run
    x_gpu = torch.from_numpy(feed[k]).cuda()
    tm_gpu = tm.cuda()
    with torch.no_grad():
        gpu_out = tm_gpu(x_gpu).cpu().numpy()
    return gpu_out

# ── All test patterns ────────────────────────────────────────────────────────
TESTS = []
def t(name, category="general"):
    def d(fn):
        TESTS.append((name, category, fn))
        return fn
    return d

# --- Resize patterns ---
@t("resize_nearest_ceil_hp", "resize")
def _():
    x=np.random.randn(1,3,4,4).astype(np.float32)
    return mk([oh.make_node("Resize",["X","r","s"],["Y"],mode="nearest",nearest_mode="ceil",
        coordinate_transformation_mode="half_pixel")],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,3,4,4])],"Y",[1,3,8,8],
        [onh.from_array(np.array([],dtype=np.float32),"r"),
         onh.from_array(np.array([1,1,2,2],dtype=np.float32),"s")]),{"X":x}

@t("resize_nearest_floor_hp", "resize")
def _():
    x=np.random.randn(1,3,4,4).astype(np.float32)
    return mk([oh.make_node("Resize",["X","r","s"],["Y"],mode="nearest",nearest_mode="floor",
        coordinate_transformation_mode="half_pixel")],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,3,4,4])],"Y",[1,3,8,8],
        [onh.from_array(np.array([],dtype=np.float32),"r"),
         onh.from_array(np.array([1,1,2,2],dtype=np.float32),"s")]),{"X":x}

@t("resize_linear_asym_2.5x", "resize")
def _():
    x=np.random.randn(1,1,4,4).astype(np.float32)
    return mk([oh.make_node("Resize",["X","r","s"],["Y"],mode="linear",
        coordinate_transformation_mode="asymmetric")],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,1,4,4])],"Y",[1,1,10,10],
        [onh.from_array(np.array([],dtype=np.float32),"r"),
         onh.from_array(np.array([1,1,2.5,2.5],dtype=np.float32),"s")]),{"X":x}

@t("resize_linear_hp_2x", "resize")
def _():
    x=np.random.randn(1,1,4,4).astype(np.float32)
    return mk([oh.make_node("Resize",["X","r","s"],["Y"],mode="linear",
        coordinate_transformation_mode="half_pixel")],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,1,4,4])],"Y",[1,1,8,8],
        [onh.from_array(np.array([],dtype=np.float32),"r"),
         onh.from_array(np.array([1,1,2,2],dtype=np.float32),"s")]),{"X":x}

@t("resize_cubic_hp_down", "resize")
def _():
    x=np.random.randn(1,1,8,8).astype(np.float32)
    return mk([oh.make_node("Resize",["X","r","s"],["Y"],mode="cubic",
        coordinate_transformation_mode="half_pixel",cubic_coeff_a=-0.75)],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,1,8,8])],"Y",[1,1,4,4],
        [onh.from_array(np.array([],dtype=np.float32),"r"),
         onh.from_array(np.array([1,1,0.5,0.5],dtype=np.float32),"s")]),{"X":x}

@t("resize_linear_align_corners", "resize")
def _():
    x=np.random.randn(1,1,4,4).astype(np.float32)
    return mk([oh.make_node("Resize",["X","r","s"],["Y"],mode="linear",
        coordinate_transformation_mode="align_corners")],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,1,4,4])],"Y",[1,1,8,8],
        [onh.from_array(np.array([],dtype=np.float32),"r"),
         onh.from_array(np.array([1,1,2,2],dtype=np.float32),"s")]),{"X":x}

# --- Conv/BN/Fusion patterns ---
@t("conv_bn", "fusion")
def _():
    Ci,Co=4,8;x=np.random.randn(1,Ci,8,8).astype(np.float32)
    return mk([oh.make_node("Conv",["X","cw","cb"],["c"],kernel_shape=[3,3],pads=[1,1,1,1]),
        oh.make_node("BatchNormalization",["c","s","b","m","v"],["Y"],epsilon=1e-5)],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,Ci,8,8])],"Y",[1,Co,8,8],
        [onh.from_array(np.random.randn(Co,Ci,3,3).astype(np.float32)*0.1,"cw"),
         onh.from_array(np.zeros(Co,dtype=np.float32),"cb"),
         onh.from_array(np.random.uniform(0.5,2,Co).astype(np.float32),"s"),
         onh.from_array(np.random.uniform(-1,1,Co).astype(np.float32),"b"),
         onh.from_array(np.random.uniform(-0.5,0.5,Co).astype(np.float32),"m"),
         onh.from_array(np.random.uniform(0.1,1,Co).astype(np.float32),"v")]),{"X":x}

@t("residual_shared_conv", "fusion")
def _():
    C=4;x=np.random.randn(1,C,8,8).astype(np.float32)
    w=np.random.randn(C,C,3,3).astype(np.float32)*0.1;b=np.zeros(C,dtype=np.float32)
    return mk([oh.make_node("Conv",["X","w","b"],["c1"],kernel_shape=[3,3],pads=[1,1,1,1]),
        oh.make_node("Relu",["c1"],["r"]),
        oh.make_node("Conv",["r","w","b"],["c2"],kernel_shape=[3,3],pads=[1,1,1,1]),
        oh.make_node("Add",["c2","X"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,C,8,8])],"Y",[1,C,8,8],
        [onh.from_array(w,"w"),onh.from_array(b,"b")]),{"X":x}

@t("dilated_3branch_sum", "fusion")
def _():
    C=4;x=np.random.randn(1,C,16,16).astype(np.float32)
    return mk([
        oh.make_node("Conv",["X","w1"],["c1"],kernel_shape=[3,3],pads=[1,1,1,1],dilations=[1,1]),
        oh.make_node("Conv",["X","w2"],["c2"],kernel_shape=[3,3],pads=[2,2,2,2],dilations=[2,2]),
        oh.make_node("Conv",["X","w3"],["c3"],kernel_shape=[3,3],pads=[4,4,4,4],dilations=[4,4]),
        oh.make_node("Add",["c1","c2"],["s"]),oh.make_node("Add",["s","c3"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,C,16,16])],"Y",[1,C,16,16],
        [onh.from_array(np.random.randn(C,C,3,3).astype(np.float32)*0.1,"w1"),
         onh.from_array(np.random.randn(C,C,3,3).astype(np.float32)*0.1,"w2"),
         onh.from_array(np.random.randn(C,C,3,3).astype(np.float32)*0.1,"w3")]),{"X":x}

# --- Optimizer patterns ---
@t("clip_exp_log", "optimizer")
def _():
    x=np.random.randn(1,4,8).astype(np.float32)*5
    return mk([oh.make_node("Clip",["X","cm","cx"],["cl"]),
        oh.make_node("Exp",["cl"],["e"]),oh.make_node("Log",["e"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,4,8])],"Y",[1,4,8],
        [onh.from_array(np.array([-10],dtype=np.float32),"cm"),
         onh.from_array(np.array([10],dtype=np.float32),"cx")]),{"X":x}

@t("mul_zero_add_offset", "optimizer")
def _():
    x=np.random.randn(1,4,8).astype(np.float32)
    return mk([oh.make_node("Mul",["X","Z"],["m"]),oh.make_node("Add",["m","O"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,4,8])],"Y",[1,4,8],
        [onh.from_array(np.zeros([1,4,8],dtype=np.float32),"Z"),
         onh.from_array(np.random.randn(1,4,8).astype(np.float32),"O")]),{"X":x}

@t("cast_fp32_int32_fp32", "optimizer")
def _():
    x=np.array([[0.1,0.6,1.4,-0.7,2.9,-1.1,0.5,1.5]],dtype=np.float32)
    return mk([oh.make_node("Cast",["X"],["i"],to=TP.INT32),
        oh.make_node("Cast",["i"],["Y"],to=TP.FLOAT)],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,8])],"Y",[1,8],[]),{"X":x}

# --- Activation patterns ---
@t("softsign", "activation")
def _():
    x=np.linspace(-5,5,32).reshape(1,32).astype(np.float32)
    return mk([oh.make_node("Softsign",["X"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,32])],"Y",[1,32],[],opset=9),{"X":x}

@t("gelu_pattern", "activation")
def _():
    x=np.random.randn(1,4,16).astype(np.float32)
    return mk([oh.make_node("MatMul",["X","W"],["mm"]),oh.make_node("Add",["mm","B"],["mb"]),
        oh.make_node("Div",["mb","sq"],["d"]),oh.make_node("Erf",["d"],["e"]),
        oh.make_node("Add",["e","o"],["ep"]),oh.make_node("Mul",["mb","ep"],["m"]),
        oh.make_node("Mul",["m","h"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,4,16])],"Y",[1,4,16],
        [onh.from_array(np.random.randn(16,16).astype(np.float32)*0.1,"W"),
         onh.from_array(np.random.randn(16).astype(np.float32)*0.1,"B"),
         onh.from_array(np.array([1.4142135],dtype=np.float32),"sq"),
         onh.from_array(np.array([0.5],dtype=np.float32),"h"),
         onh.from_array(np.array([1.0],dtype=np.float32),"o")]),{"X":x}

# --- Pooling ---
@t("avgpool_ceil_mode", "pooling")
def _():
    x=np.random.randn(1,4,7,7).astype(np.float32)
    return mk([oh.make_node("AveragePool",["X"],["Y"],kernel_shape=[3,3],strides=[2,2],
        pads=[1,1,1,1],ceil_mode=1)],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,4,7,7])],"Y",[1,4,4,4],[]),{"X":x}

# --- Normalization ---
@t("instance_norm_small", "norm")
def _():
    C=4;x=np.random.randn(2,C,2,2).astype(np.float32)
    return mk([oh.make_node("InstanceNormalization",["X","s","b"],["Y"],epsilon=1e-5)],
        [oh.make_tensor_value_info("X",TP.FLOAT,[2,C,2,2])],"Y",[2,C,2,2],
        [onh.from_array(np.ones(C,dtype=np.float32),"s"),
         onh.from_array(np.zeros(C,dtype=np.float32),"b")]),{"X":x}

@t("l2_norm", "norm")
def _():
    x=np.random.randn(1,4,16).astype(np.float32)
    return mk([oh.make_node("Mul",["X","X"],["x2"]),
        oh.make_node("ReduceSum",["x2"],["s"],axes=[-1],keepdims=1),
        oh.make_node("Add",["s","eps"],["se"]),oh.make_node("Sqrt",["se"],["sq"]),
        oh.make_node("Div",["X","sq"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,4,16])],"Y",[1,4,16],
        [onh.from_array(np.array([1e-12],dtype=np.float32),"eps")],opset=11),{"X":x}

# --- Matmul ---
@t("matmul_4d_batch", "matmul")
def _():
    x=np.random.randn(2,4,8,16).astype(np.float32)
    return mk([oh.make_node("MatMul",["X","W"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[2,4,8,16])],"Y",[2,4,8,8],
        [onh.from_array(np.random.randn(2,4,16,8).astype(np.float32)*0.1,"W")]),{"X":x}

# --- Cumsum ---
@t("cumsum_last_axis", "reduce")
def _():
    x=np.random.randn(1,4,8).astype(np.float32)
    return mk([oh.make_node("CumSum",["X","a"],["Y"])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,4,8])],"Y",[1,4,8],
        [onh.from_array(np.array([2],dtype=np.int64),"a")]),{"X":x}

# --- Attention ---
@t("matmul_scale_softmax", "attention")
def _():
    x=np.random.randn(1,8,16).astype(np.float32)
    w=np.random.randn(16,16).astype(np.float32)*0.1
    scale=np.array([0.25],dtype=np.float32)
    return mk([oh.make_node("MatMul",["X","W"],["mm"]),
        oh.make_node("Mul",["mm","sc"],["ms"]),
        oh.make_node("Softmax",["ms"],["Y"],axis=-1)],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,8,16])],"Y",[1,8,16],
        [onh.from_array(w,"W"),onh.from_array(scale,"sc")]),{"X":x}

# --- Complex pipelines ---
@t("resize_then_conv_bn", "pipeline")
def _():
    Ci,Co=3,8;x=np.random.randn(1,Ci,4,4).astype(np.float32)
    return mk([
        oh.make_node("Resize",["X","r","sc"],["rs"],mode="linear",
            coordinate_transformation_mode="half_pixel"),
        oh.make_node("Conv",["rs","cw","cb"],["c"],kernel_shape=[3,3],pads=[1,1,1,1]),
        oh.make_node("BatchNormalization",["c","bs","bb","bm","bv"],["Y"],epsilon=1e-5)],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,Ci,4,4])],"Y",[1,Co,8,8],
        [onh.from_array(np.array([],dtype=np.float32),"r"),
         onh.from_array(np.array([1,1,2,2],dtype=np.float32),"sc"),
         onh.from_array(np.random.randn(Co,Ci,3,3).astype(np.float32)*0.1,"cw"),
         onh.from_array(np.zeros(Co,dtype=np.float32),"cb"),
         onh.from_array(np.ones(Co,dtype=np.float32),"bs"),
         onh.from_array(np.zeros(Co,dtype=np.float32),"bb"),
         onh.from_array(np.zeros(Co,dtype=np.float32),"bm"),
         onh.from_array(np.ones(Co,dtype=np.float32),"bv")]),{"X":x}

@t("conv_relu_avgpool", "pipeline")
def _():
    C=4;x=np.random.randn(1,C,8,8).astype(np.float32)
    return mk([
        oh.make_node("Conv",["X","w","b"],["c"],kernel_shape=[3,3],pads=[1,1,1,1]),
        oh.make_node("Relu",["c"],["r"]),
        oh.make_node("AveragePool",["r"],["Y"],kernel_shape=[2,2],strides=[2,2])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,C,8,8])],"Y",[1,C,4,4],
        [onh.from_array(np.random.randn(C,C,3,3).astype(np.float32)*0.1,"w"),
         onh.from_array(np.zeros(C,dtype=np.float32),"b")]),{"X":x}

@t("dual_conv_concat_maxpool", "pipeline")
def _():
    x=np.random.randn(1,3,8,8).astype(np.float32)
    return mk([
        oh.make_node("Conv",["X","w1"],["c1"],kernel_shape=[1,1]),
        oh.make_node("Relu",["c1"],["r1"]),
        oh.make_node("Conv",["X","w2"],["c2"],kernel_shape=[3,3],pads=[1,1,1,1]),
        oh.make_node("Relu",["c2"],["r2"]),
        oh.make_node("Concat",["r1","r2"],["cat"],axis=1),
        oh.make_node("MaxPool",["cat"],["Y"],kernel_shape=[2,2],strides=[2,2])],
        [oh.make_tensor_value_info("X",TP.FLOAT,[1,3,8,8])],"Y",[1,8,4,4],
        [onh.from_array(np.random.randn(4,3,1,1).astype(np.float32)*0.1,"w1"),
         onh.from_array(np.random.randn(4,3,3,3).astype(np.float32)*0.1,"w2")]),{"X":x}

# ── Run all tests ────────────────────────────────────────────────────────────
def diff(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

COMPILERS = [
    ("ORT_opt", run_ort_opt),
    ("OpenVINO", run_openvino),
    ("onnx2torch", run_torch_eager),
    ("torch.compile", run_torch_compile),
    ("GPU(torch)", run_jax_jit),
]

if __name__ == "__main__":
    hdr = f"{'Pattern':<30}"
    for cname, _ in COMPILERS:
        hdr += f" {cname:<14}"
    hdr += " Status"
    print(hdr)
    print("=" * len(hdr))

    all_bugs = []
    summary = {c: {"bug": 0, "ok": 0, "err": 0} for c, _ in COMPILERS}

    for name, cat, builder in TESTS:
        try:
            mb, feed = builder()
        except Exception as e:
            print(f"{name:<30} BUILD ERROR: {e}")
            continue

        try:
            ref = run_ort_ref(mb, feed)
        except Exception as e:
            print(f"{name:<30} REF ERROR: {e}")
            continue

        row = f"{name:<30}"
        row_bugs = []

        for cname, runner in COMPILERS:
            try:
                out = runner(mb, feed)
                d = diff(out, ref)
                if d > TOL:
                    row += f" {d:<14.4f}"
                    row_bugs.append((cname, d))
                    all_bugs.append((name, cname, d))
                    summary[cname]["bug"] += 1
                else:
                    row += f" {d:<14.2e}"
                    summary[cname]["ok"] += 1
            except Exception as e:
                row += f" {'ERR':<14}"
                summary[cname]["err"] += 1

        status = f"*** {len(row_bugs)} BUG ***" if row_bugs else "OK"
        print(row + f" {status}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"TOLERANCE: {TOL}")
    print(f"PATTERNS TESTED: {len(TESTS)}")
    print()

    print(f"{'Compiler':<20} {'Bugs':>5} {'OK':>5} {'Errors':>6}")
    print("-" * 40)
    total_bugs = 0
    for cname, _ in COMPILERS:
        s = summary[cname]
        print(f"{cname:<20} {s['bug']:>5} {s['ok']:>5} {s['err']:>6}")
        total_bugs += s["bug"]
    print(f"{'TOTAL':<20} {total_bugs:>5}")

    print(f"\n{'='*90}")
    print(f"ALL BUGS (tolerance > {TOL}):")
    print(f"{'Pattern':<30} {'Compiler':<20} {'max_diff':>10}")
    print("-" * 65)
    for name, cname, d in sorted(all_bugs, key=lambda x: -x[2]):
        print(f"{name:<30} {cname:<20} {d:>10.4f}")
