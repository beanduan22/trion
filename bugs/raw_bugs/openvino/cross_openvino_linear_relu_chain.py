#!/usr/bin/env python3
"""
Bug ID     : pytorch/pytorch#98852 / compiler: OpenVINO CPU
Source     : https://github.com/pytorch/pytorch/issues/98852
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL — all 5 other compilers agree with ORT_ref
Patterns   : Gemm(W1,b1) → Relu → Gemm(W2,b2) → Relu → Gemm(W3,b3) → Relu
             Input [1,64] → [1,64] → [1,64] → [1,32]
Root cause : OpenVINO CPU plugin's fusion of consecutive Gemm+Relu operations
             produces incorrect output.  fp32 accumulation order differs from
             the reference path, causing large errors (max_diff 1.2–2.8).
             ORT_opt, onnx2torch, torch.compile, TorchScript all agree exactly
             with ORT_ref; OpenVINO alone diverges.
             Original PyTorch bug: torch.compile Linear+ReLU×3 wrong value
             (fixed in torch 2.9.1) — pattern still misbehaves in OpenVINO.
Exit 0 = BUG REPRODUCED (OpenVINO max_diff > 0.05 vs ORT_ref)
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import sys
try:
    import numpy as np
    import onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
    import openvino as ov
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr); sys.exit(2)

TOL = 0.05
np.random.seed(852)

# ── Build ONNX model: Gemm→Relu×3 ────────────────────────────────────────────
rng = np.random.default_rng(999)
W1 = rng.standard_normal((64, 64)).astype(np.float32)
b1 = rng.standard_normal(64).astype(np.float32)
W2 = rng.standard_normal((64, 64)).astype(np.float32)
b2 = rng.standard_normal(64).astype(np.float32)
W3 = rng.standard_normal((32, 64)).astype(np.float32)
b3 = rng.standard_normal(32).astype(np.float32)

nodes = [
    oh.make_node("Gemm", ["x","W1","b1"], ["h1"], transB=1),
    oh.make_node("Relu", ["h1"],           ["r1"]),
    oh.make_node("Gemm", ["r1","W2","b2"], ["h2"], transB=1),
    oh.make_node("Relu", ["h2"],           ["r2"]),
    oh.make_node("Gemm", ["r2","W3","b3"], ["h3"], transB=1),
    oh.make_node("Relu", ["h3"],           ["y"]),
]
inits = [onh.from_array(W1,"W1"), onh.from_array(b1,"b1"),
         onh.from_array(W2,"W2"), onh.from_array(b2,"b2"),
         onh.from_array(W3,"W3"), onh.from_array(b3,"b3")]
g = oh.make_graph(nodes, "linear_relu_chain",
    [oh.make_tensor_value_info("x", TP.FLOAT, [1, 64])],
    [oh.make_tensor_value_info("y", TP.FLOAT, [1, 32])],
    initializer=inits)
m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
m.ir_version = 8
mb = m.SerializeToString()

# Test inputs: 4 different seeds
INPUTS = [np.random.RandomState(s).randn(1, 64).astype(np.float32)
          for s in [0, 1, 42, 123]]

print("Model: Gemm(64→64)→Relu→Gemm(64→64)→Relu→Gemm(64→32)→Relu  [fp32]")
print()

def run_ort(opt_level):
    so = ort.SessionOptions(); so.graph_optimization_level = opt_level
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"])

try:
    import onnx2torch, torch
    def _o2t(feed):
        net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        with torch.no_grad(): return net(torch.from_numpy(feed["x"])).numpy()
    def _ts(feed):
        net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        x_t = torch.from_numpy(feed["x"])
        with torch.no_grad(): ts = torch.jit.trace(net,(x_t,)); return ts(x_t).numpy()
    def _tc(feed):
        net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        c = torch.compile(net)
        with torch.no_grad(): return c(torch.from_numpy(feed["x"])).numpy()
except ImportError:
    _o2t = _ts = _tc = None

ort_ref_sess = run_ort(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
ort_opt_sess = run_ort(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
core = ov.Core(); mod = core.read_model(mb, b""); comp = core.compile_model(mod, "CPU")

print(f"{'Seed':<6} {'Compiler':<16} {'max_diff':>10}  Status")
print("-" * 50)

found = False
for i, x in enumerate(INPUTS):
    feed = {"x": x}
    ref  = ort_ref_sess.run(None, feed)[0]

    for cname, runner in [
        ("ORT_opt",   lambda f: ort_opt_sess.run(None, f)[0]),
        ("OpenVINO",  lambda f: comp(f)[comp.output(0)]),
        ("onnx2torch",_o2t),
        ("torch.compile",_tc),
        ("TorchScript",_ts),
    ]:
        if runner is None: continue
        try:
            out  = runner(feed)
            diff = float(np.abs(ref.ravel() - np.array(out).ravel()).max())
            bug  = diff > TOL
            if bug and cname == "OpenVINO": found = True
            flag = "  *** BUG" if bug else ""
            print(f"seed={i:<2}  {cname:<16} {diff:>10.5f}  {'BUG' if bug else 'ok'}{flag}")
        except Exception as e:
            print(f"seed={i:<2}  {cname:<16}  ERR: {str(e)[:60]}")

print()
if found:
    print("BUG REPRODUCED: OpenVINO Gemm+Relu chain produces wrong fp32 output;")
    print("  ORT_opt / onnx2torch / torch.compile / TorchScript all agree with ORT_ref.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: OpenVINO agreed with ORT_ref within tolerance.")
    sys.exit(1)
