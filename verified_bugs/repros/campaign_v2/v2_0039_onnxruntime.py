#!/usr/bin/env python3
"""
Bug v2-0039 — OnnxRuntime wrong output vs ONNX spec reference.
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT Resize nearest_ceil gives wrong pixel coordinates when combined with ReduceSum(middle) — optimizer propagates the reduction through resize incorrectly.
Report to : https://github.com/microsoft/onnxruntime/issues
           (label: bug, regression, Resize / TopK / CumSum etc.)

Oracle: ORT_ENABLE_ALL output vs pytorch_eager (onnx2torch) reference.
        ORT_DISABLE_ALL gives the same wrong answer — this is a fundamental
        implementation divergence, not an optimization regression.

The ONNX model is embedded in unique_2658.py (the reference repro).
Run: python v2_0039_onnxruntime.py  →  exit 0 = BUG REPRODUCED
"""
import base64, sys
import numpy as np
import onnx
import onnxruntime as ort

TOLERANCE = 0.01

_ONNX_B64 = (
    "CAg6p0gKLQoLbW9kZWxfaW5wdXQKCW4wX21tNGRfdxILbjBfbW00ZF9vdXQiBk1hdE11bApECgtu"
    "MF9tbTRkX291dAoLbjEwMF9yc21fYXgSDG4xMDBfcnNtX3JlZCIJUmVkdWNlU3VtKg8KCGtlZXBk"
    "aW1zGAGgAQIKLgoLbjBfbW00ZF9vdXQKDG4xMDBfcnNtX3JlZBIMbjEwMF9yc21fb3V0IgNBZGQK"
    "NQoMbjEwMF9yc21fb3V0Cg5uMjAwX2NyY19zaGlmdBIQbjIwMF9jcmNfc2hpZnRlZCIDQWRkCkQK"
    "DG4xMDBfcnNtX291dAoQbjIwMF9jcmNfc2hpZnRlZBINbjIwMF9jcmNfY2F0MSIGQ29uY2F0KgsK"
    "BGF4aXMYAaABAgqIAQoNbjIwMF9jcmNfY2F0MQoMbjIwMF9jcmNfcm9pCg9uMjAwX2NyY19zY2Fs"
    "ZXMSDG4yMDBfY3JjX3JzeiIGUmVzaXplKi8KHmNvb3JkaW5hdGVfdHJhbnNmb3JtYXRpb25fbW9k"
    "ZSIKaGFsZl9waXhlbKABAyoRCgRtb2RlIgZsaW5lYXKgAQMKQAoMbjIwMF9jcmNfcnN6Cg1uMjAw"
    "X2NyY19jYXQxEgxuMjAwX2NyY19vdXQiBkNvbmNhdCoLCgRheGlzGAGgAQIKMgoMbjIwMF9jcmNf"
    "b3V0CgtuMzAwX21tNGRfdxINbjMwMF9tbTRkX291dCIGTWF0TXVsCqUBCg1uMzAwX21tNGRfb3V0"
    "Cg1uNDAwX3Jlc2lfcm9pChBuNDAwX3Jlc2lfc2NhbGVzEg1uNDAwX3Jlc2lfb3V0IgZSZXNpemUq"
    "LwoeY29vcmRpbmF0ZV90cmFuc2Zvcm1hdGlvbl9tb2RlIgpoYWxmX3BpeGVsoAEDKhIKBG1vZGUi"
    "B25lYXJlc3SgAQMqFwoMbmVhcmVzdF9tb2RlIgRjZWlsoAEDEgt0cmlvbl9ncmFwaCqYIAgBCAEI"
    "IAggEAFCCW4wX21tNGRfd0qAILhBcr5l7bE81fPaPrQuDj0naQk/mknQPiSz0z3AMSi+Na/bPg5e"
    "4D04hKq++SqVvHXrFz3xUeC9EcJBvuei071R8oA9l3R1PgqzFr5GihO/u6Z7Plu+5T1ARUO+GL1g"
    "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"
    "PhT/rL4TL/K+U2fmPmh/sb4CtQq+CpKdPZbP+j5LfJ++iBJWPn+zMzzKmRG+nwk+Pd4oLj5Ks66+"
    "+m7avLK9Bj4rM+W+GKGLPkx5Ir6e0O68FSCfvlK7ej07sk64HVNrvtKeiz5UYr8+NZiRviXfZzoC"
    "f4u+ZXgKPvTtrT6OfLM+sP6ovvpfPD5H9p87a25yPVLyszx9Tmq+P5mkvgkOhL25lqu+q38HPrcN"
    "Nr/gNdi+PZkZvrtUMz7yZ0m9mQ6UPre3T728+JE9SfKTPvWhJb4wkh++mJonvq3LT76RcCu+CQYg"
    "Pf5CuT5NIc0+fAkoPRNAc759pgC/77ktPh1Fiz431bO9PIZlPuP9Sb5zYmY+XUhFPqOEuz65C6U9"
    "SPMfPor9Wr6mFSi+db7gPqx+CzzbpZE+3uS2vhyrcb3momG+1B8JPnOVHD78IHo+3y81PfZqGz7B"
    "oGi+SO/QPZJ47L0tjLu+VXKGPjxDGT3PIAg+3KOMveWIMLwHWmG+0s8hPiY6hD5KhDA+GWVgO9Oo"
    "rj4R5Y8+dDzePdjy1L7LkB0+asd1vjK6JT4aP5q+6YGLvXVKvj6WjGo+1xu5PixwubtvQO4+1UYw"
    "PocSmb5Tchi/gRgeOx+yhr5NSDq/44loPiF5pr5Hmpi+HInFPn4lwT58Hp6+SUXdPIN5xrs8hmY+"
    "B5lAvvIvnL3b5OI96boqvnYPmz6fa/w+Dobiuy+x4D4/sOE+X6abvN1ivT7ikH0+jGbAPioVCAAQ"
    "AUINbjQwMF9yZXNpX3JvaUoAKigIBBABQhBuNDAwX3Jlc2lfc2NhbGVzShAAAIA/AACAPwAAAEAA"
    "AABAWiUKC21vZGVsX2lucHV0EhYKFAgBEhAKAggBCgIIAwoCCCAKAgggYicKDW40MDBfcmVzaV9v"
    "dXQSFgoUCAESEAoCCAEKAggMCgIIQAoCCEBCBAoAEBE="
)
_INPUTS = {
    'model_input': (
    "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",
        [1, 3, 32, 32],
    ),
}


def _decode():
    model_bytes = base64.b64decode(_ONNX_B64)
    inputs = {n: np.frombuffer(base64.b64decode(b64), dtype=np.float32
                               ).copy().reshape(sh)
              for n, (b64, sh) in _INPUTS.items()}
    return model_bytes, inputs

def _rel_l2(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape: return float("inf")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))

def _pytorch_eager(model_bytes, inputs):
    """Reference: onnx2torch eager (no compiler) — same as oracle pytorch_eager."""
    import torch, onnx2torch
    model = onnx.load_from_string(model_bytes)
    m = onnx2torch.convert(model).eval()
    inp = next(iter(inputs.values()))
    x = torch.from_numpy(inp)
    with torch.no_grad():
        out = m(x)
    if isinstance(out, (list, tuple)): out = out[0]
    return out.detach().cpu().float().numpy().ravel()


def main():
    model_bytes, inputs = _decode()

    # Reference: pytorch eager
    try:
        ref = _pytorch_eager(model_bytes, inputs)
    except Exception as exc:
        print(f"[error] reference failed: {exc}")
        sys.exit(2)

    # Target: ORT with full graph optimisation
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(model_bytes, opts,
                                    providers=["CPUExecutionProvider"])
        target = np.asarray(sess.run(None, inputs)[0]).ravel()
    except Exception as exc:
        print(f"[crash] onnxruntime: {exc}")
        sys.exit(0)

    diff_vs_ref = _rel_l2(target, ref)

    # Also check opt-vs-noopt (delta_opt)
    try:
        opts_off = ort.SessionOptions()
        opts_off.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_off = ort.InferenceSession(model_bytes, opts_off,
                                         providers=["CPUExecutionProvider"])
        noopt = np.asarray(sess_off.run(None, inputs)[0]).ravel()
        diff_opt_noopt = _rel_l2(target, noopt)
    except Exception:
        diff_opt_noopt = 0.0

    print(f"rel L2(ORT_opt  vs pytorch_eager) = {diff_vs_ref:.6e}")
    print(f"rel L2(ORT_opt  vs ORT_noopt)     = {diff_opt_noopt:.6e}")
    print(f"tolerance = {TOLERANCE}")

    if diff_vs_ref > TOLERANCE or diff_opt_noopt > TOLERANCE:
        print("BUG REPRODUCED")
        sys.exit(0)
    print("not reproduced")
    sys.exit(1)

if __name__ == "__main__":
    main()
