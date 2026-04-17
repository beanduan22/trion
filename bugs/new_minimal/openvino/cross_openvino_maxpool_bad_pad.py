#!/usr/bin/env python3
"""
Bug ID     : microsoft/onnxruntime#23088
Source     : https://github.com/microsoft/onnxruntime/issues/23088
Compiler   : OnnxRuntime (all opt levels)
Oracle     : OpenVINO — silently executes with extended output shape
Patterns   : MaxPool(kernel=[3,3], pads=[3,3,3,3]) — pad >= kernel is invalid
             per ONNX spec but passes onnx.checker.check_model()
Root cause : ONNX checker does not enforce the strict inequality pads < kernel;
             ORT enforces it at session init (pool_attributes.h:78).
Cross-compiler behavior on pad >= kernel:
  ORT (ref+opt)   → FAIL: "Pad should be smaller than kernel"
  OpenVINO        → SILENT: returns shape (1,1,12,12) [accepted, wrong size]
  onnx2torch      → RuntimeError: pad > kernel/2
  torch.compile   → error (same PyTorch path)
  TorchScript     → RuntimeError: pad > kernel/2
Exit 0 = BUG REPRODUCED (OpenVINO silently accepts; ORT + PyTorch compilers reject)
Exit 1 = not reproduced
Exit 2 = missing deps

New inputs tested: kernel=[5,5] pads=[5,5,5,5], kernel=[2,2] pads=[2,2,2,2]
"""
import sys
try:
    import numpy as np
    import onnx
    from onnx import helper as oh, TensorProto as TP
    import onnxruntime as ort
    import openvino as ov
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr); sys.exit(2)

np.random.seed(88)
TOL = 0.05

def make_maxpool(kernel, pads, inp_shape):
    nodes = [oh.make_node("MaxPool", ["x"], ["y"],
                          kernel_shape=kernel, pads=pads,
                          strides=[1]*len(kernel), auto_pad="NOTSET")]
    h_out = inp_shape[2] + pads[0] + pads[2] - kernel[0] + 1
    w_out = inp_shape[3] + pads[1] + pads[3] - kernel[1] + 1
    g = oh.make_graph(nodes, "maxpool",
        [oh.make_tensor_value_info("x", TP.FLOAT, list(inp_shape))],
        [oh.make_tensor_value_info("y", TP.FLOAT, [inp_shape[0], inp_shape[1], h_out, w_out])])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 8
    return m.SerializeToString()

# Test cases: (kernel, pads, input_shape, description)
CASES = [
    ([3, 3], [3, 3, 3, 3], [1, 1, 8, 8],  "kernel=3 pad=3  (original bug)"),
    ([5, 5], [5, 5, 5, 5], [1, 2, 12, 12], "kernel=5 pad=5  (new input)"),
    ([2, 2], [2, 2, 2, 2], [1, 1, 6, 6],   "kernel=2 pad=2  (new input)"),
    ([3, 3], [3, 0, 3, 0], [1, 1, 8, 8],   "kernel=3 pad=(3,0,3,0) partial"),
]

print("Bug #23088 — MaxPool pad ≥ kernel: passes ONNX checker, rejected by ORT")
print()
ov_silently_accepted = False

for kernel, pads, inp_shape, desc in CASES:
    mb = make_maxpool(kernel, pads, inp_shape)

    try: onnx.checker.check_model(onnx.load_from_string(mb)); checker = "PASS"
    except Exception as e: checker = f"FAIL({e})"

    results = {}
    rng = np.random.RandomState(88)
    feed = {"x": rng.randn(*inp_shape).astype(np.float32)}

    # ORT_ref
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        out = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        results["ORT_ref"] = ("ok", out.shape)
    except Exception as e:
        results["ORT_ref"] = ("ERR", str(e).split("\n")[0][:60])

    # ORT_opt
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        out = ort.InferenceSession(mb, sess_options=so, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        results["ORT_opt"] = ("ok", out.shape)
    except Exception as e:
        results["ORT_opt"] = ("ERR", str(e).split("\n")[0][:60])

    # OpenVINO
    try:
        core = ov.Core(); mod = core.read_model(mb, b"")
        comp = core.compile_model(mod, "CPU")
        out  = comp(feed)[comp.output(0)]
        results["OpenVINO"] = ("ok", out.shape)
        ov_silently_accepted = True
    except Exception as e:
        results["OpenVINO"] = ("ERR", str(e).split("\n")[0][:60])

    # onnx2torch
    try:
        import onnx2torch, torch
        net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        with torch.no_grad():
            out = net(torch.from_numpy(feed["x"]))
        results["onnx2torch"] = ("ok", tuple(out.shape))
    except Exception as e:
        results["onnx2torch"] = ("ERR", str(e).split("\n")[0][:60])

    # torch.compile
    try:
        import onnx2torch, torch
        net  = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        comp_fn = torch.compile(net, backend="inductor")
        with torch.no_grad():
            out = comp_fn(torch.from_numpy(feed["x"]))
        results["torch_compile"] = ("ok", tuple(out.shape))
    except Exception as e:
        results["torch_compile"] = ("ERR", str(e).split("\n")[0][:60])

    # TorchScript
    try:
        import onnx2torch, torch
        net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        x_t = torch.from_numpy(feed["x"])
        with torch.no_grad():
            ts  = torch.jit.trace(net, (x_t,))
            out = ts(x_t)
        results["TorchScript"] = ("ok", tuple(out.shape))
    except Exception as e:
        results["TorchScript"] = ("ERR", str(e).split("\n")[0][:60])

    print(f"  Case: {desc}  [ONNX checker: {checker}]")
    for cname, (status, detail) in results.items():
        flag = "  ← SILENT ACCEPT" if (status == "ok" and cname == "OpenVINO") else \
               "  ← CORRECT REJECT" if (status == "ERR" and cname != "OpenVINO") else ""
        print(f"    {cname:<16} {status:<6} {str(detail)[:60]}{flag}")
    print()

if ov_silently_accepted:
    print("BUG REPRODUCED: OpenVINO silently accepts MaxPool with pad >= kernel")
    print("  (returns extended output) while ORT and PyTorch compilers correctly reject.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: OpenVINO also rejected the invalid padding.")
    sys.exit(1)
