#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_inception_v3_branch
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Multiple Conv branches (1x1, 3x3, 5x5) -> Concat  [Inception module]
Root cause : OV CPU Conv fp32 tiling differs from ORT; multi-branch accumulates.
Tolerance  : 0.01

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
x  = np.random.randn(1, 32, 16, 16).astype(np.float32)
W1 = np.random.randn(32, 32, 1, 1).astype(np.float32)
W3 = np.random.randn(32, 32, 3, 3).astype(np.float32)
W5 = np.random.randn(16, 32, 5, 5).astype(np.float32)
Wp = np.random.randn(16, 32, 1, 1).astype(np.float32)

nodes = [
    oh.make_node("Conv", ["X","W1"], ["b1"], kernel_shape=[1,1]),
    oh.make_node("Conv", ["X","W3"], ["b3"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("Conv", ["X","W5"], ["b5"], kernel_shape=[5,5], pads=[2,2,2,2]),
    oh.make_node("MaxPool", ["X"],   ["pool"], kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]),
    oh.make_node("Conv", ["pool","Wp"], ["bp"], kernel_shape=[1,1]),
    oh.make_node("Concat", ["b1","b3","b5","bp"], ["Y"], axis=1),
]
graph = oh.make_graph(nodes, "inception",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,32,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,96,16,16])],
    initializer=[
        onh.from_array(W1,"W1"), onh.from_array(W3,"W3"),
        onh.from_array(W5,"W5"), onh.from_array(Wp,"Wp"),
    ])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
mb = model.SerializeToString()

# ── Multi-backend comparison ─────────────────────────────────────────────────
FEED = {"X": x}
TOL  = 0.01

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
    print(f"BUG REPRODUCED: OpenVINO diverges from ORT_ref (tol=0.01).")
    sys.exit(0)
print("NOT REPRODUCED")
sys.exit(1)
