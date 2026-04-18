#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_conv_bn_eval_explicit
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : Conv(3x3) -> BatchNormalization (eval mode, explicit params)
Root cause : OV CPU Conv+BN fusion fp32 rounding differs from ORT unoptimized.
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
C_IN, C_OUT = 32, 64
x      = np.random.randn(1, C_IN, 16, 16).astype(np.float32)
W      = np.random.randn(C_OUT, C_IN, 3, 3).astype(np.float32)
b      = np.zeros(C_OUT, dtype=np.float32)
scale  = np.random.uniform(0.5, 2.0, C_OUT).astype(np.float32)
bshift = np.random.uniform(-1.0, 1.0, C_OUT).astype(np.float32)
mean   = np.random.uniform(-0.5, 0.5, C_OUT).astype(np.float32)
var    = np.random.uniform(0.1, 1.0, C_OUT).astype(np.float32)

nodes = [
    oh.make_node("Conv", ["X","W","cb"], ["conv"], kernel_shape=[3,3], pads=[1,1,1,1]),
    oh.make_node("BatchNormalization", ["conv","sc","bs","bm","bv"], ["Y"], epsilon=1e-5),
]
graph = oh.make_graph(nodes, "test",
    [oh.make_tensor_value_info("X", TP.FLOAT, [1,C_IN,16,16])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [1,C_OUT,16,16])],
    initializer=[
        onh.from_array(W,"W"), onh.from_array(b,"cb"),
        onh.from_array(scale,"sc"), onh.from_array(bshift,"bs"),
        onh.from_array(mean,"bm"), onh.from_array(var,"bv"),
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
