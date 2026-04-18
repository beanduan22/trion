#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_fp16_matmul_add
Source     : Cross-compiler testing (2026-04-14)
Compiler   : OpenVINO 2026.0, CPU plugin
Patterns   : fp16 MatMul [1,N] × [N,N] + fp16 bias [N]
Root cause : OpenVINO CPU plugin's fp16 tiled GEMM accumulates partial sums
             in a different order than ORT's sequential fp16 path.  fp16 lacks
             the precision to make addition associative at scale, so tile-level
             reordering produces a different result.  Error grows with N:
               N=32  → 0.027   N=64  → 0.078   N=128 → 0.188
             This is the same root cause as ORT#23284 but manifests in OV.
Tolerance  : 0.05

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

np.random.seed(0)
N = 64
W = np.random.randn(N, N).astype(np.float16)
B = np.random.randn(N).astype(np.float16)
x = np.random.randn(1, N).astype(np.float16)

graph = helper.make_graph(
    [helper.make_node("MatMul", ["x", "W"], ["mm"]),
     helper.make_node("Add",    ["mm", "B"], ["y"])],
    "fp16_matmul_add",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, N])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, N])],
    initializer=[numpy_helper.from_array(W, "W"),
                 numpy_helper.from_array(B, "B")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
model_bytes = model.SerializeToString()
mb = model_bytes

# ORT reference (no optimisation)

# ── Multi-backend comparison ─────────────────────────────────────────────────
FEED = {"x": x}
TOL  = 0.05

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
    print(f"BUG REPRODUCED: OpenVINO diverges from ORT_ref (tol=0.05).")
    _sys.exit(0)
print("NOT REPRODUCED")
__sys.exit(1)
