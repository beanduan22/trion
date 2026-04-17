#!/usr/bin/env python3
"""
Bug ID     : tf_61865 / compiler: onnx2torch + PyTorch eager
Source     : https://github.com/tensorflow/tensorflow/issues/61865
Compiler   : onnx2torch (ONNX → PyTorch nn.Module, eager execution)
Patterns   : Gather(params[8], indices=[2,5,4,7,8,3]) — index 8 is OOB for size-8 tensor
Root cause : onnx2torch lowers ONNX Gather to direct PyTorch indexing, which raises
             IndexError on OOB.  OpenVINO returns 0.0 silently; TF-XLA clamps silently.
             Compiler comparison:
               ORT_ref / ORT_opt → INVALID_ARGUMENT
               OpenVINO          → [1,1,1,1,0,1]  (silent 0.0)
               onnx2torch        → IndexError      ← TARGET
               torch.compile     → runtime error
               TorchScript       → IndexError
               TF-XLA (original) → [1,1,1,1,1,1]  (silent clamp)
Exit 0 = BUG REPRODUCED (onnx2torch errors while other compilers return values)
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
    import openvino as ov
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

np.random.seed(3)

# ── Build ONNX model ─────────────────────────────────────────────────────────
# New input: params size=6 with random values; index 6 is OOB for size-6
params  = np.array([2., 4., 6., 8., 10., 12.], dtype=np.float32)
indices = np.array([0, 2, 4, 5, 6, 1], dtype=np.int64)    # index 6 is OOB

nodes = [oh.make_node("Gather", ["params", "indices"], ["Y"], axis=0)]
inits = [onh.from_array(indices, "indices")]
g = oh.make_graph(nodes, "gather_oob",
    [oh.make_tensor_value_info("params", TP.FLOAT, [6])],
    [oh.make_tensor_value_info("Y",      TP.FLOAT, [6])],
    initializer=inits)
m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
m.ir_version = 8
mb   = m.SerializeToString()
feed = {"params": params}

print(f"params  : {params.tolist()}")
print(f"indices : {indices.tolist()}  (index 6 is OOB for size-6 tensor)")
print()

# ── Run all compilers ─────────────────────────────────────────────────────────
results = {}

def record(name, fn):
    try:
        results[name] = (fn(), None)
    except Exception as e:
        results[name] = (None, str(e).split("\n")[0][:90])

def _ort_ref():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"]).run(None, feed)[0]
record("ORT_ref", _ort_ref)

def _ort_opt():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"]).run(None, feed)[0]
record("ORT_opt", _ort_opt)

def _openvino():
    core = ov.Core()
    mod  = core.read_model(mb, b"")
    comp = core.compile_model(mod, "CPU")
    return comp(feed)[comp.output(0)]
record("OpenVINO", _openvino)

def _onnx2torch():
    net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    with torch.no_grad():
        out = net(torch.from_numpy(params))
    return (out[0] if isinstance(out, (list, tuple)) else out).numpy()
record("onnx2torch", _onnx2torch)

def _torch_compile():
    net  = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    comp = torch.compile(net, backend="inductor")
    with torch.no_grad():
        out = comp(torch.from_numpy(params))
    return (out[0] if isinstance(out, (list, tuple)) else out).numpy()
record("torch_compile", _torch_compile)

def _torchscript():
    net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    x_t = torch.from_numpy(params)
    with torch.no_grad():
        ts  = torch.jit.trace(net, (x_t,))
        out = ts(x_t)
    return (out[0] if isinstance(out, (list, tuple)) else out).numpy()
record("TorchScript", _torchscript)

# ── Report ────────────────────────────────────────────────────────────────────
print(f"{'Compiler':<16}  {'Output':<32}  Error")
print("-" * 80)
for name, (out, err) in results.items():
    out_str = str(out.tolist()) if out is not None else "—"
    err_str = err[:55] if err else "—"
    flag    = "  ← TARGET" if name == "onnx2torch" else ""
    print(f"{name:<16}  {out_str:<32}  {err_str}{flag}")

# ── Verdict ───────────────────────────────────────────────────────────────────
o2t_out, o2t_err = results.get("onnx2torch", (None, "not run"))
others_with_value = {k: results[k][0].tolist() for k in results
                     if k != "onnx2torch" and results[k][0] is not None}

print()
if o2t_err and others_with_value:
    print(f"BUG REPRODUCED: onnx2torch raises ({o2t_err[:60]})")
    print(f"  while other compilers return values: {others_with_value}")
    sys.exit(0)
elif not o2t_err:
    print("NOT REPRODUCED: onnx2torch returned a value (bounds check removed or fixed).")
    sys.exit(1)
else:
    print("NOT REPRODUCED: all compilers raised errors.")
    sys.exit(1)
