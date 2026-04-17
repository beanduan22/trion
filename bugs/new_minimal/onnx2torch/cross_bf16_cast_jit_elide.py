#!/usr/bin/env python3
"""
Bug ID     : cross_bf16_cast_jit_elide
Source     : Cross-compiler testing — onnx2torch converter
Compiler   : onnx2torch (ONNX -> torch.nn graph) vs ORT / torch reference
Oracle     : bf16 round-trip ground truth via torch `.to(bfloat16).to(float32)`
             on the same MatMul weight.  Also compared against ORT when the
             installed ORT build supports bf16 Cast / MatMul opsets.
Patterns   : Cast(float32 -> bfloat16) -> Cast(bfloat16 -> float32) -> MatMul(W)
             — an intentional, *lossy* bf16 precision-truncation pattern used
             as a quantization-aware-training probe.
Root cause : onnx2torch's op-graph lowering folds the adjacent Cast pair to
             an identity (or preserves fp32 across the bf16 intermediate)
             instead of materialising the bf16 rounding.  The MatMul that
             follows then operates on full-precision fp32 rather than on
             bf16-truncated activations, producing a numerically different
             result from ORT / the torch bf16 ground truth.
Tolerance  : 1e-4 absolute on the matmul output — bf16 truncation of a
             MatMul input produces O(1e-3) perturbations on typical weights,
             well above this tolerance, so any elision is visible.

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime as ort
    import torch
    import onnx2torch
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

TOL = 1e-4
np.random.seed(0)

# Build ONNX: X(fp32) -> Cast(bf16) -> Cast(fp32) -> MatMul(W) -> Y
N, K, M = 1, 8, 4
X_np = np.random.randn(N, K).astype(np.float32)
W_np = np.random.randn(K, M).astype(np.float32)

X_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, K])
Y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, M])
W_init = numpy_helper.from_array(W_np, name="W")

nodes = [
    helper.make_node("Cast", ["X"], ["X_bf16"], to=TensorProto.BFLOAT16, name="cast_down"),
    helper.make_node("Cast", ["X_bf16"], ["X_fp32_rt"], to=TensorProto.FLOAT, name="cast_up"),
    helper.make_node("MatMul", ["X_fp32_rt", "W"], ["Y"], name="matmul"),
]
graph = helper.make_graph(nodes, "bf16_roundtrip", [X_vi], [Y_vi], initializer=[W_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
model_bytes = model.SerializeToString()

# Ground truth: torch bf16 round-trip + matmul in fp32 weights
x_t = torch.from_numpy(X_np)
w_t = torch.from_numpy(W_np)
x_truncated = x_t.to(torch.bfloat16).to(torch.float32)
gt = (x_truncated @ w_t).detach().numpy()
print(f"torch bf16 ground truth first 8 : {gt.ravel()[:8]}")

# Sanity: confirm bf16 truncation actually differs from fp32 matmul
fp32_matmul = (x_t @ w_t).detach().numpy()
gt_vs_fp32 = float(np.max(np.abs(gt - fp32_matmul)))
print(f"|gt - fp32_no_trunc| max_abs   : {gt_vs_fp32:.6g}  (must be >> tol)")
if gt_vs_fp32 < TOL:
    print("bf16 truncation indistinguishable from fp32 on this input — regenerate")
    sys.exit(1)

# Try ORT (may or may not support bf16 Cast / MatMul)
ort_out = None
try:
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"X": X_np})[0]
    print(f"ORT output first 8             : {ort_out.ravel()[:8]}")
    ort_diff = float(np.max(np.abs(ort_out - gt)))
    print(f"|ORT - gt| max_abs             : {ort_diff:.6g}")
except Exception as e:
    print(f"ORT skipped (no bf16 op support): {type(e).__name__}: {str(e)[:120]}")

# Now convert via onnx2torch
try:
    onnx_model = onnx.load_from_string(model_bytes)
    torch_module = onnx2torch.convert(onnx_model).eval()
except Exception as e:
    print(f"onnx2torch conversion crashed: {type(e).__name__}: {str(e)[:200]}")
    # Conversion crash on a valid bf16 round-trip graph is itself a bug
    print("BUG REPRODUCED: onnx2torch cannot materialise Cast(fp32->bf16)->Cast(bf16->fp32)")
    sys.exit(0)

with torch.no_grad():
    try:
        o2t_out = torch_module(x_t).detach().numpy()
    except Exception as e:
        print(f"onnx2torch forward crashed: {type(e).__name__}: {str(e)[:200]}")
        print("BUG REPRODUCED: onnx2torch module fails to execute bf16 round-trip graph")
        sys.exit(0)

print(f"onnx2torch output first 8      : {o2t_out.ravel()[:8]}")
diff_gt = float(np.max(np.abs(o2t_out - gt)))
diff_fp32 = float(np.max(np.abs(o2t_out - fp32_matmul)))
print(f"|onnx2torch - gt|   max_abs    : {diff_gt:.6g}")
print(f"|onnx2torch - fp32| max_abs    : {diff_fp32:.6g}")

# Reproduction: onnx2torch result is closer to fp32-matmul (no truncation) than
# to bf16 ground truth — or it diverges from bf16 ground truth by > tolerance.
if diff_gt > TOL:
    print(f"\nBUG REPRODUCED: onnx2torch diverges from bf16 ground truth by {diff_gt:.6g} > {TOL}")
    if diff_fp32 < diff_gt:
        print("  onnx2torch output matches *fp32* matmul more closely — Cast pair was elided.")
    sys.exit(0)

print("\nnot reproduced — onnx2torch materialises bf16 truncation within tolerance")
sys.exit(1)
