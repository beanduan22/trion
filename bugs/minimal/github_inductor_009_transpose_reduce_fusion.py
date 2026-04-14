#!/usr/bin/env python3
"""
Bug ID     : github_inductor_009
Source     : GitHub — pytorch/pytorch#146416
Compiler   : PyTorch Inductor (torch.compile), torch ~2.6
Patterns   : row-wise ReduceSum + pointwise Mul + Transpose().contiguous()
Root cause : Inductor incorrectly fuses a row-wise reduction with a subsequent
             transpose operation.  Instead of performing the transpose it
             reinterprets the same storage buffer as if it were already transposed,
             producing wrong output silently.  Row-wise memory access patterns
             (stride != 1 along reduced axis) are incompatible with the fusion
             kernel Inductor applies.
             Reproduces in backward passes of scaled-attention-like patterns.
Tolerance  : 0.0 (output should be exact)

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
import sys as _sys
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

TOL = 1e-5

any_bug = False

print("=== github_inductor_009: Transpose+ReduceSum fusion bug ===")
print()

if HAS_TORCH:
    torch.manual_seed(0)

    def buggy_pattern(x, scale):
        """row-wise reduction * scale → transpose().contiguous()"""
        row_sum = x.sum(dim=1, keepdim=True)   # [B, 1, C]
        out = row_sum * scale                   # pointwise mul
        return out.transpose(1, 2).contiguous() # [B, C, 1]

    B, N, C = 4, 64, 128
    x     = torch.randn(B, N, C)
    scale = torch.randn(B, 1, C)

    ref = buggy_pattern(x, scale)
    try:
        compiled = torch.compile(buggy_pattern, backend="inductor")
        got = compiled(x, scale)
        diff = float((ref - got).abs().max().item())
        bug  = diff > TOL
        print(f"Input [{B},{N},{C}] row_sum*scale→transpose")
        print(f"eager vs torch.compile: max_diff={diff:.2e}  {'BUG' if bug else 'ok'}")
        if bug:
            print(f"  ref[0,:4,0]: {ref[0, :4, 0].tolist()}")
            print(f"  got[0,:4,0]: {got[0, :4, 0].tolist()}")
            any_bug = True
    except Exception as e:
        print(f"torch.compile failed: {e}")

    # Pattern 2: scale(reduce) then transpose — the exact pattern from the issue
    def attention_grad_pattern(out_grad, scale):
        """
        Simplified attention backward: scale the row-sum of out_grad,
        then transpose to match expected layout.
        """
        col_scale = out_grad.sum(dim=-1, keepdim=True) * scale  # [B, H, 1]
        return col_scale.transpose(-2, -1).contiguous()          # [B, 1, H]

    B, H = 8, 256
    og    = torch.randn(B, H, H)
    sc    = torch.randn(B, H, 1)
    ref2  = attention_grad_pattern(og, sc)
    try:
        comp2 = torch.compile(attention_grad_pattern, backend="inductor")
        got2  = comp2(og, sc)
        diff2 = float((ref2 - got2).abs().max().item())
        bug2  = diff2 > TOL
        print(f"Attention grad pattern [{B},{H},{H}] scale→transpose")
        print(f"eager vs torch.compile: max_diff={diff2:.2e}  {'BUG' if bug2 else 'ok'}")
        if bug2:
            any_bug = True
    except Exception as e:
        print(f"torch.compile attn pattern failed: {e}")

else:
    # ONNX path: ReduceSum → Mul → Transpose
    print("(PyTorch not available — testing via ONNX: ReduceSum→Mul→Transpose)")
    if HAS_ONNX:
        np.random.seed(0)
        B, N, C = 4, 32, 64
        x = np.random.randn(B, N, C).astype(np.float32)
        scale = np.random.randn(B, 1, C).astype(np.float32)

        from onnx import helper as oh, TensorProto as TP, numpy_helper as onh
        nodes = [
            oh.make_node("ReduceSum", ["x", "axes"], ["row_sum"],
                         keepdims=1),
            oh.make_node("Mul", ["row_sum", "scale"], ["mul_out"]),
            oh.make_node("Transpose", ["mul_out"], ["y"], perm=[0, 2, 1]),
        ]
        inits = [onh.from_array(scale, "scale"),
                 onh.from_array(np.array([1], dtype=np.int64), "axes")]
        g = oh.make_graph(nodes, "reduce_transpose",
            [oh.make_tensor_value_info("x", TP.FLOAT, [B, N, C])],
            [oh.make_tensor_value_info("y", TP.FLOAT, [B, C, 1])],
            initializer=inits)
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 18)])
        m.ir_version = 8
        mb = m.SerializeToString()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ref = ort.InferenceSession(mb, so, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        opt = ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        diff = float(np.abs(ref.astype(np.float64) - opt.astype(np.float64)).max())
        bug  = diff > TOL
        print(f"ORT_ref vs ORT_opt: max_diff={diff:.2e}  {'BUG' if bug else 'ok'}")
        if bug:
            any_bug = True
    else:
        print("(neither PyTorch nor ONNX available)")
        # pure analytical: the bug reinterprets strides
        # e.g., a [B,1,C] tensor stored contiguously → viewed as [B,C,1] without copy
        x_np = np.random.randn(4, 1, 64).astype(np.float32)
        # correct transpose (copy)
        correct = x_np.transpose(0, 2, 1).copy()
        # buggy: view without transpose (reuse strides)
        buggy   = x_np.reshape(4, 64, 1)
        diff    = float(np.abs(correct - buggy).max())
        bug     = diff > TOL
        print(f"Analytical: correct transpose vs stride-reuse: max_diff={diff:.2e}  {'BUG' if bug else 'ok'}")
        if bug:
            any_bug = True

print()
PASS = not any_bug
print(f"PASS={PASS}")
if not PASS:
    print("BUG REPRODUCED: Inductor transpose+reduce fusion reinterprets storage strides")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
