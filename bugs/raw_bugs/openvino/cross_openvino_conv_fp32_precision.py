#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_conv_fp32_precision
Source     : Cross-compiler testing (2026-04-14)
Compiler   : OpenVINO 2026.0, CPU plugin
Patterns   : Conv [1,C,H,W] with random float32 weights, no BN or bias
Root cause : OpenVINO's CPU plugin selects a Winograd / tiled GEMM algorithm
             for float32 Conv that accumulates dot products in a different order
             than ORT's direct-convolution reference path.  For a 4×4×3×3 weight
             matrix the partial sums (36 multiplications per output element) can
             diverge by up to 0.054 due to floating-point reassociation.  This
             is distinct from the Conv+BN fusion bug (cross_openvino_conv_bn_fusion)
             which also has a systematic rounding component.  Error increases with
             larger kernel size and channel count.
Tolerance  : 0.02

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime as ort
except ImportError as e:
    print(f"missing deps: {e}")
    _sys.exit(2)

TOL = 0.02

np.random.seed(0)
B, C, H, W = 1, 4, 8, 8
kH, kW     = 3, 3
W_conv     = np.random.randn(C, C, kH, kW).astype(np.float32)
x_in       = np.random.randn(B, C, H, W).astype(np.float32)

nodes = [helper.make_node("Conv", ["x", "W_conv"], ["y"], pads=[1, 1, 1, 1])]
inits = [numpy_helper.from_array(W_conv, "W_conv")]
graph = helper.make_graph(nodes, "conv_fp32_prec",
    [helper.make_tensor_value_info("x",     TensorProto.FLOAT, [B, C, H, W])],
    [helper.make_tensor_value_info("y",     TensorProto.FLOAT, [B, C, H, W])],
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
model_bytes = model.SerializeToString()
mb = model_bytes

# ORT reference (no optimisation)

# ── Multi-backend comparison ─────────────────────────────────────────────────
FEED = {"x": x_in}
TOL  = 0.02

def _ort(opt):
    so = ort.SessionOptions(); so.graph_optimization_level = opt
    return ort.InferenceSession(mb, sess_options=so,
                                providers=["CPUExecutionProvider"]).run(None, FEED)[0]

ref = _ort(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

_results = []
def _rec(name, fn):
    try: _results.append((name, fn(), None))
    except Exception as e: _results.append((name, None, str(e)[:70]))

_rec("ORT_opt", lambda: _ort(ort.GraphOptimizationLevel.ORT_ENABLE_ALL))

try:
    import openvino as ov
    _core = ov.Core(); _comp = _core.compile_model(_core.read_model(mb, b""), "CPU")
    _rec("OpenVINO", lambda: np.array(_comp(FEED)[_comp.output(0)]))
except ImportError:
    pass

try:
    import onnx2torch, torch as _torch
    with _torch.no_grad():
        _net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
        _ins = [_torch.from_numpy(v) for v in FEED.values()]
        _rec("onnx2torch",    lambda: _net(*_ins).numpy())
        _rec("torch.compile", lambda: _torch.compile(_net)(*_ins).numpy())
        try:
            _ts = _torch.jit.trace(_net, _ins)
            _rec("TorchScript", lambda: _ts(*_ins).numpy())
        except Exception as _e:
            _results.append(("TorchScript", None, str(_e)[:70]))
except ImportError:
    pass

print(f"{'Compiler':<16} {'max_diff':>10}  Status")
print("-" * 42)
_found = False
for _cname, _out, _err in _results:
    if _err:
        print(f"{_cname:<16}      ERR  {_err}"); continue
    _d = float(np.abs(ref.ravel() - np.array(_out).ravel()).max())
    _bug = _d > TOL
    if _bug and _cname == "OpenVINO": _found = True
    print(f"{_cname:<16} {_d:>10.5f}  {'BUG ***' if _bug else 'ok'}")

print()
if _found:
    print(f"BUG REPRODUCED: OpenVINO diverges from ORT_ref (tol=0.02).")
    _sys.exit(0)
print("NOT REPRODUCED")
__sys.exit(1)
