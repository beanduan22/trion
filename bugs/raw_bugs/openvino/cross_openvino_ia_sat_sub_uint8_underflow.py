#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_ia_sat_sub_uint8_underflow
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Cast(float->uint8) -> Clip (simulating uint8 subtraction underflow)
Root cause : OV may handle integer arithmetic near boundaries differently from ORT.
             When uint8 sub would underflow, OV saturates (clamps to 0) vs wrapping.
Tolerance  : checks saturation vs exact behavior

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys
try:
    import numpy as np, onnx
    from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
    import onnxruntime as ort
except ImportError as e:
    print(f"missing dep: {e}"); sys.exit(2)

np.random.seed(42)
# Values that after Cast to float and Sub produce values near 0 boundary
x = np.array([5.0, 3.0, 10.0, 1.0, 2.0, 8.0, 200.0, 100.0], dtype=np.float32).reshape(2,4)
# Sub with constant where result should be ~0 or slightly negative
sub_c = np.array([5.5, 3.5, 10.5, 1.5, 2.5, 8.5, 200.5, 100.5], dtype=np.float32).reshape(2,4)
cmin = np.array(0.0, dtype=np.float32)

nodes = [
    oh.make_node("Sub",  ["X","SC"],       ["diff"]),
    oh.make_node("Relu", ["diff"],         ["relu_out"]),  # saturating behavior: floor at 0
    # OV may fuse Sub+Relu differently than ORT for near-boundary values
    oh.make_node("Add",  ["relu_out","X"], ["Y"]),          # expose the difference
]
graph = oh.make_graph(nodes, "sat_sub",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,4])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,4])],
    initializer=[onh.from_array(sub_c,"SC")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

# ── Multi-backend comparison ─────────────────────────────────────────────────
FEED = {"X": x}
TOL  = 0.001

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
    print(f"BUG REPRODUCED: OpenVINO diverges from ORT_ref (tol=0.001).")
    sys.exit(0)
print("NOT REPRODUCED")
sys.exit(1)
