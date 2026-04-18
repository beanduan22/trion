#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_matmul_add_biasgelu_bcast
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL
Patterns   : MatMul(x,W) -> Add(bias) -> GELU -> Mul(gate)  [biased GELU gate]
Root cause : OV CPU GEMM fp32 accumulation differs; GELU nonlinearity amplifies.
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
x    = np.random.randn(2, 512).astype(np.float32)
W    = np.random.randn(512, 64).astype(np.float32)
b    = np.random.randn(64).astype(np.float32)
gate = np.random.randn(2, 64).astype(np.float32)

# Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x+0.044715*x^3)))
# Build as: mm+b -> x*0.5*(1+tanh(c1*(x+c2*x^3)))
c1 = np.array(0.7978845608, dtype=np.float32)
c2 = np.array(0.044715, dtype=np.float32)
c05 = np.array(0.5, dtype=np.float32)
c1v = np.array(1.0, dtype=np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W"],        ["mm"]),
    oh.make_node("Add",    ["mm","b"],        ["h"]),
    # GELU approx
    oh.make_node("Mul",    ["h","h"],         ["h2"]),
    oh.make_node("Mul",    ["h2","h"],        ["h3"]),
    oh.make_node("Mul",    ["h3","c2"],       ["h3c"]),
    oh.make_node("Add",    ["h","h3c"],       ["hsum"]),
    oh.make_node("Mul",    ["hsum","c1"],     ["harg"]),
    oh.make_node("Tanh",   ["harg"],          ["htanh"]),
    oh.make_node("Add",    ["htanh","c1v"],   ["h1p"]),
    oh.make_node("Mul",    ["h1p","c05"],     ["hgelu"]),
    oh.make_node("Mul",    ["h","hgelu"],     ["gelu_out"]),
    oh.make_node("Mul",    ["gelu_out","G"],  ["Y"]),
]
graph = oh.make_graph(nodes, "biasgelu",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[
        onh.from_array(W,"W"), onh.from_array(b,"b"), onh.from_array(gate,"G"),
        onh.from_array(c1,"c1"), onh.from_array(c2,"c2"),
        onh.from_array(c05,"c05"), onh.from_array(c1v,"c1v"),
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
