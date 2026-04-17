#!/usr/bin/env python3
"""
Bug ID     : pytorch/pytorch#109925 / compilers: onnx2torch, torch.compile, TorchScript
Source     : https://github.com/pytorch/pytorch/issues/109925
Compiler   : onnx2torch + PyTorch-based backends (torch.compile, TorchScript)
Oracle     : ORT_DISABLE_ALL — ORT_opt and OpenVINO also agree
Patterns   : Cast(Input[1,16], INT64) → CumSum(axis=1) → output[1,16]
             Input values are 0/1 (boolean-like).
Root cause : onnx2torch lowers ONNX CumSum to torch.cumsum.  When the input
             dtype is INT64 (cast from bool-like float), PyTorch treats it as
             a bool tensor in certain code paths and returns wrong cumulative
             values.  ORT correctly accumulates integer values.
             Example: input [1,0,1,1,...] → ORT gives [1,1,2,3,...],
             torch.cumsum gives wrong integer accumulation.
             max_diff ranges from 5 to 10 depending on seed.
             ORT_opt and OpenVINO are NOT affected.
             Distinct from cross_onnx2torch_cumsum.py (that bug is about dynamic
             axis causing a graph break; this bug is about wrong INT64 accumulation).
Exit 0 = BUG REPRODUCED (onnx2torch / torch.compile / TorchScript differ from ORT_ref)
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import sys
try:
    import numpy as np
    import onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
    import onnx2torch
    import torch
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr); sys.exit(2)

TOL = 0.05

# ── Build ONNX model: Cast(float→INT64) → CumSum ─────────────────────────────
# Cast float input to INT64 first so CumSum operates on integer type
nodes = [
    oh.make_node("Cast",    ["x"],        ["xi"], to=TP.INT64),
    oh.make_node("CumSum",  ["xi", "ax"], ["y"]),
]
inits = [onh.from_array(np.array(1, dtype=np.int64), "ax")]   # axis=1
g = oh.make_graph(nodes, "cumsum_bool",
    [oh.make_tensor_value_info("x",  TP.FLOAT, [1, 16])],
    [oh.make_tensor_value_info("y",  TP.INT64, [1, 16])],
    initializer=inits)
m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
m.ir_version = 8
mb = m.SerializeToString()

# Test inputs: 0/1 float values (boolean-like)
SEEDS = [0, 1, 42, 123]
def make_feed(seed):
    rng = np.random.RandomState(seed)
    return {"x": rng.randint(0, 2, size=(1, 16)).astype(np.float32)}

print("Model: Cast(float→INT64) → CumSum(axis=1)  input=[1,16] bool-like 0/1 values")
print()

def run_ort(opt_level, feed):
    so = ort.SessionOptions(); so.graph_optimization_level = opt_level
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"]).run(None, feed)[0]

def run_openvino(feed):
    import openvino as ov
    core = ov.Core(); mod = core.read_model(mb, b"")
    comp = core.compile_model(mod, "CPU")
    return comp(feed)[comp.output(0)]

def run_onnx2torch(feed):
    net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    with torch.no_grad():
        out = net(torch.from_numpy(feed["x"]))
    return (out[0] if isinstance(out,(list,tuple)) else out).numpy()

def run_torchscript(feed):
    net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    x_t = torch.from_numpy(feed["x"])
    with torch.no_grad():
        ts  = torch.jit.trace(net, (x_t,))
        out = ts(x_t)
    return (out[0] if isinstance(out,(list,tuple)) else out).numpy()

def run_torch_compile(feed):
    net  = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    comp = torch.compile(net, backend="inductor")
    with torch.no_grad():
        out = comp(torch.from_numpy(feed["x"]))
    return (out[0] if isinstance(out,(list,tuple)) else out).numpy()

RUNNERS = [
    ("ORT_opt",       lambda f: run_ort(ort.GraphOptimizationLevel.ORT_ENABLE_ALL, f)),
    ("OpenVINO",      run_openvino),
    ("onnx2torch",    run_onnx2torch),
    ("torch_compile", run_torch_compile),
    ("TorchScript",   run_torchscript),
]

print(f"{'Seed':<6} {'Compiler':<16} {'max_diff':>10}  Status")
print("-" * 50)

found = False
for seed in SEEDS:
    feed = make_feed(seed)
    ref  = run_ort(ort.GraphOptimizationLevel.ORT_DISABLE_ALL, feed)
    print(f"seed={seed:<3}  input={feed['x'][0,:6].astype(int).tolist()}...  ref={ref[0,:6].tolist()}...")
    for cname, runner in RUNNERS:
        try:
            out  = runner(feed)
            diff = float(np.abs(ref.ravel().astype(np.float64) -
                                np.array(out).ravel().astype(np.float64)).max())
            bug  = diff > TOL
            if bug: found = True
            flag = "  *** BUG" if bug else ""
            print(f"       {cname:<16} {diff:>10.4f}  {'BUG' if bug else 'ok'}{flag}")
        except Exception as e:
            print(f"       {cname:<16}  ERR: {str(e)[:70]}")
    print()

if found:
    print("BUG REPRODUCED: onnx2torch / torch.compile / TorchScript give wrong INT64 CumSum;")
    print("  ORT_ref, ORT_opt, OpenVINO all compute the correct cumulative sum.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: all compilers agreed with ORT_ref.")
    sys.exit(1)
