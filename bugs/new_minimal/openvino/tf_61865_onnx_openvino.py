#!/usr/bin/env python3
"""
Bug ID     : tf_61865 / compiler: OpenVINO CPU
Source     : https://github.com/tensorflow/tensorflow/issues/61865
Compiler   : OpenVINO CPU
Patterns   : Gather(params[8], indices=[2,5,4,7,8,3]) — index 8 is OOB for size-8 tensor
Root cause : OpenVINO silently returns 0.0 for the OOB slot instead of raising an error.
             All other compilers either error or (TF-XLA) clamp to the last element.
             Compiler comparison:
               ORT_ref / ORT_opt → INVALID_ARGUMENT
               OpenVINO          → [1,1,1,1,0,1]  (OOB → 0.0, silent)  ← TARGET
               onnx2torch        → IndexError
               torch.compile     → runtime error
               TorchScript       → IndexError
               TF-XLA (original) → [1,1,1,1,1,1]  (OOB → clamp, silent)
Exit 0 = BUG REPRODUCED (OpenVINO returns value silently while others error)
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
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)

np.random.seed(2)

# ── Build ONNX model ─────────────────────────────────────────────────────────
# New input: params size=12 with distinct values; index 12 is OOB for size-12
params  = np.arange(1, 13, dtype=np.float32)   # [1,2,...,12]
indices = np.array([0, 3, 6, 9, 12, 1], dtype=np.int64)   # index 12 is OOB

nodes = [oh.make_node("Gather", ["params", "indices"], ["Y"], axis=0)]
inits = [onh.from_array(indices, "indices")]
g = oh.make_graph(nodes, "gather_oob",
    [oh.make_tensor_value_info("params", TP.FLOAT, [12])],
    [oh.make_tensor_value_info("Y",      TP.FLOAT, [6])],
    initializer=inits)
m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
m.ir_version = 8
mb   = m.SerializeToString()
feed = {"params": params}

print(f"params  : {params.tolist()}")
print(f"indices : {indices.tolist()}  (index 12 is OOB for size-12 tensor)")
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

try:
    import onnx2torch, torch
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
except ImportError:
    for k in ("onnx2torch", "torch_compile", "TorchScript"):
        results[k] = (None, "missing dep")

# ── Report ────────────────────────────────────────────────────────────────────
print(f"{'Compiler':<16}  {'Output':<32}  Error")
print("-" * 80)
for name, (out, err) in results.items():
    out_str = str(out.tolist()) if out is not None else "—"
    err_str = err[:55] if err else "—"
    flag    = "  ← TARGET" if name == "OpenVINO" else ""
    print(f"{name:<16}  {out_str:<32}  {err_str}{flag}")

# ── Verdict ───────────────────────────────────────────────────────────────────
ov_out, ov_err = results.get("OpenVINO", (None, "not run"))
ort_errors = all(results.get(k, (None, None))[1] is not None
                 for k in ("ORT_ref", "ORT_opt"))

print()
if ov_out is not None and ort_errors:
    oob_val = float(ov_out[4])
    print("BUG REPRODUCED: OpenVINO silently returns a value for OOB Gather")
    print(f"  OOB slot (index 12) → {oob_val:.4f}")
    print(f"  ORT_ref/ORT_opt raise INVALID_ARGUMENT; OpenVINO proceeds silently.")
    sys.exit(0)
elif ov_err:
    print(f"NOT REPRODUCED: OpenVINO raised: {ov_err}")
    sys.exit(1)
else:
    print("NOT REPRODUCED: ORT also returned a value (strict validation relaxed).")
    sys.exit(1)
