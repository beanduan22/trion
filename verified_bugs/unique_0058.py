#!/usr/bin/env python3
"""
Bug #0058 — onnxruntime (ORT_ENABLE_ALL) diverges from pytorch_eager.

Patterns  : [['normalization', 'reduce_mean_first_axis'], ['broadcast', 'cumsum_last_axis'], ['attention', 'matmul_4d_batch'], ['attention', 'matmul_4d_batch'], ['constant', 'cast_fp32_int32_roundtrip'], ['broadcast', 'neg_unary']]
S_diff    : 1.000000   (ORT_opt vs pytorch_eager)
Delta_opt : 0.000000   (ORT_opt vs ORT_noopt)
Tolerance : 0.001

    python unique_0058.py
Exit 0 = reproduced, 1 = not reproduced.
"""
import base64, sys
import numpy as np
import onnx
import onnxruntime as ort

TOLERANCE = 0.001

_ONNX_B64 = (
    "CAg650UKQwoLbW9kZWxfaW5wdXQSCm4wX3JtZl9yZWQiClJlZHVjZU1lYW4qCwoEYXhlc0AAoAEH"
    "Kg8KCGtlZXBkaW1zGAGgAQIKKgoLbW9kZWxfaW5wdXQKCm4wX3JtZl9yZWQSCm4wX3JtZl9vdXQi"
    "A0FkZAotCgpuMF9ybWZfb3V0CgpuMTAwX2NzX2F4EgtuMTAwX2NzX291dCIGQ3VtU3VtCjEKC24x"
    "MDBfY3Nfb3V0CgtuMjAwX21tNGRfdxINbjIwMF9tbTRkX291dCIGTWF0TXVsCjMKDW4yMDBfbW00"
    "ZF9vdXQKC24zMDBfbW00ZF93Eg1uMzAwX21tNGRfb3V0IgZNYXRNdWwKIgoNbjMwMF9tbTRkX291"
    "dBIMbjQwMF9jZmlyX2FiIgNBYnMKIgoMbjQwMF9jZmlyX2FiEgxuNDAwX2NmaXJfY2UiBENlaWwK"
    "LQoMbjQwMF9jZmlyX2NlEgxuNDAwX2NmaXJfY2kiBENhc3QqCQoCdG8YBqABAgotCgxuNDAwX2Nm"
    "aXJfY2kSDG40MDBfY2Zpcl9jZiIEQ2FzdCoJCgJ0bxgBoAECCjEKDG40MDBfY2Zpcl9jZgoNbjMw"
    "MF9tbTRkX291dBINbjQwMF9jZmlyX291dCIDTXVsCiEKDW40MDBfY2Zpcl9vdXQSC241MDBfbmVn"
    "X19hIgNOZWcKMAoLbjUwMF9uZWdfX2EKDW40MDBfY2Zpcl9vdXQSDW41MDBfbmVnX19vdXQiA011"
    "bBILdHJpb25fZ3JhcGgqGBAHQgpuMTAwX2NzX2F4SggDAAAAAAAAACqaIAgBCAEIIAggEAFCC24y"
    "MDBfbW00ZF93SoAguEFyvmXtsTzV89o+tC4OPSdpCT+aSdA+JLPTPcAxKL41r9s+Dl7gPTiEqr75"
    "KpW8desXPfFR4L0RwkG+56LTvVHygD2XdHU+CrMWvkaKE7+7pns+W77lPUBFQ74YvWC+bTu3Pjgv"
    "Ar9gLY49lIKFPsdU1T3l5um8wlUGPr69Ib74vIq+5lmrvnjSEj1kKsi+gcpkPsElr77kFKc9zHWa"
    "vq+cdLzsHMO9oiSOPmF39j5/ns28RswjPuK0fr0ecy8+qKDfPQUHt73EujE+4/5GPgSxMT5Y/b89"
    "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"
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
