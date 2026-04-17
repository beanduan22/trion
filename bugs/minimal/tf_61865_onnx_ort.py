#!/usr/bin/env python3
"""
Bug ID     : tf_61865 / compiler: OnnxRuntime (ORT_DISABLE_ALL + ORT_ENABLE_ALL)
Source     : https://github.com/tensorflow/tensorflow/issues/61865
Compiler   : OnnxRuntime (ORT_DISABLE_ALL and ORT_ENABLE_ALL)
Patterns   : Gather(params[8], indices=[2,5,4,7,8,3]) — index 8 is OOB for size-8 tensor
Root cause : ORT strictly validates index bounds and raises INVALID_ARGUMENT for any
             out-of-bounds index, even when the ONNX spec leaves OOB Gather undefined.
             Other compilers handle OOB differently:
               ORT_ref / ORT_opt → INVALID_ARGUMENT (this file)
               OpenVINO          → silent 0.0
               onnx2torch        → IndexError
               torch.compile     → runtime error
               TorchScript       → IndexError
               TF-XLA (original) → silent clamp to last element (1.0)
Exit 0 = BUG REPRODUCED (ORT errors while at least one other compiler returns a value)
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import sys

try:
    import numpy as np
    import onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

np.random.seed(1)

# ── Build ONNX model: Gather with one OOB index ───────────────────────────────
# New input: params size=5 with distinct values; index 5 is OOB for size-5 tensor
params  = np.array([10., 20., 30., 40., 50.], dtype=np.float32)
indices = np.array([0, 1, 2, 3, 5, 4], dtype=np.int64)    # index 5 is OOB

nodes = [oh.make_node("Gather", ["params", "indices"], ["Y"], axis=0)]
inits = [onh.from_array(indices, "indices")]
g = oh.make_graph(nodes, "gather_oob",
    [oh.make_tensor_value_info("params", TP.FLOAT, [5])],
    [oh.make_tensor_value_info("Y",      TP.FLOAT, [6])],
    initializer=inits)
m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
m.ir_version = 8
mb   = m.SerializeToString()
feed = {"params": params}

print(f"params  : {params.tolist()}")
print(f"indices : {indices.tolist()}  (index 5 is OOB for size-5 tensor)")
print()

# ── Run all compilers ─────────────────────────────────────────────────────────
results = {}

def record(name, fn):
    try:
        results[name] = (fn(), None)
    except Exception as e:
        results[name] = (None, str(e).split("\n")[0][:90])

# ORT_ref
def _ort_ref():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"]).run(None, feed)[0]
record("ORT_ref", _ort_ref)

# ORT_opt
def _ort_opt():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"]).run(None, feed)[0]
record("ORT_opt", _ort_opt)

# OpenVINO
try:
    import openvino as ov
    def _openvino():
        core = ov.Core()
        mod  = core.read_model(mb, b"")
        comp = core.compile_model(mod, "CPU")
        return comp(feed)[comp.output(0)]
    record("OpenVINO", _openvino)
except ImportError:
    results["OpenVINO"] = (None, "missing dep")

# onnx2torch
try:
    import onnx2torch, torch
    def _onnx2torch():
        net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        with torch.no_grad():
            out = net(torch.from_numpy(params))
        return (out[0] if isinstance(out, (list, tuple)) else out).numpy()
    record("onnx2torch", _onnx2torch)
except ImportError:
    results["onnx2torch"] = (None, "missing dep")

# torch.compile
try:
    import onnx2torch, torch
    def _torch_compile():
        net  = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        comp = torch.compile(net, backend="inductor")
        with torch.no_grad():
            out = comp(torch.from_numpy(params))
        return (out[0] if isinstance(out, (list, tuple)) else out).numpy()
    record("torch_compile", _torch_compile)
except ImportError:
    results["torch_compile"] = (None, "missing dep")

# TorchScript
try:
    import onnx2torch, torch
    def _torchscript():
        net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        x_t = torch.from_numpy(params)
        with torch.no_grad():
            ts  = torch.jit.trace(net, (x_t,))
            out = ts(x_t)
        return (out[0] if isinstance(out, (list, tuple)) else out).numpy()
    record("TorchScript", _torchscript)
except ImportError:
    results["TorchScript"] = (None, "missing dep")

# ── Report ────────────────────────────────────────────────────────────────────
print(f"{'Compiler':<16}  {'Output':<32}  Error")
print("-" * 80)
for name, (out, err) in results.items():
    out_str = str(out.tolist()) if out is not None else "—"
    err_str = err[:55] if err else "—"
    flag    = "  ← TARGET" if name in ("ORT_ref", "ORT_opt") else ""
    print(f"{name:<16}  {out_str:<32}  {err_str}{flag}")

# ── Verdict ───────────────────────────────────────────────────────────────────
ort_errors   = all(results.get(k, (None, None))[1] is not None
                   for k in ("ORT_ref", "ORT_opt"))
others_value = any(results.get(k, (None, None))[0] is not None
                   for k in ("OpenVINO", "onnx2torch", "torch_compile", "TorchScript"))

print()
if ort_errors and others_value:
    vals = {k: results[k][0].tolist() for k in results
            if results[k][0] is not None}
    print("BUG REPRODUCED: ORT (ref + opt) raise INVALID_ARGUMENT on OOB Gather")
    print(f"  while other compilers return values: {vals}")
    sys.exit(0)
elif not ort_errors:
    print("NOT REPRODUCED: ORT returned a value (strict validation relaxed or bug fixed).")
    sys.exit(1)
else:
    print("NOT REPRODUCED: all compilers raised errors.")
    sys.exit(1)
