#!/usr/bin/env python3
"""
Bug ID     : cross_openvino_sub_self_mul_zero
Compiler   : OpenVINO CPU
Oracle     : ORT_DISABLE_ALL (CPU EP)
Patterns   : MatMul fp32 accumulation diff with subtractive cancellation.
             Graph = Mul( Sub( MatMul(X,W1), MatMul(X,W2) ), const )
             where W1 ≠ W2. Not a true self-subtraction — we rely on the
             cancellation between two close-magnitude GEMMs to amplify any
             order-of-accumulation differences between OpenVINO's CPU GEMM
             kernel and ORT's reference GEMM.
Why it's informative:
             With random 4×512 × 512×128 inputs both matmul products are
             O(√512) ≈ 22 per element. Taking their difference destroys most
             of the signal, so any per-kernel fp32 reordering (e.g. AVX2
             tile blocking vs scalar order) shows up unmasked in the
             low-order bits. Multiplying by 10.0 then brings the delta
             above a generous 0.01 tolerance if the backends really do
             accumulate in different orders.
Note       : Historically this check was labelled "sub_self should give
             near-zero". That is incorrect for this graph (W1 ≠ W2); we've
             relabelled it to reflect what it actually tests — a
             cancellation-sensitive GEMM cross-check between OpenVINO CPU
             and ORT CPU.
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
x  = np.random.randn(4, 512).astype(np.float32)
W1 = np.random.randn(512, 128).astype(np.float32)
W2 = np.random.randn(512, 128).astype(np.float32)
# Sub of two different MatMul results (not self-sub) - still shows fp32 diff
sc = np.array(10.0, dtype=np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W1"],   ["mm1"]),
    oh.make_node("MatMul", ["X","W2"],   ["mm2"]),
    oh.make_node("Sub",    ["mm1","mm2"], ["diff"]),
    oh.make_node("Mul",    ["diff","sc"], ["Y"]),
]
graph = oh.make_graph(nodes, "sub_mul",
    [oh.make_tensor_value_info("X", TP.FLOAT, [4,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [4,128])],
    initializer=[onh.from_array(W1,"W1"), onh.from_array(W2,"W2"), onh.from_array(sc,"sc")])
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
