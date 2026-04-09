#!/usr/bin/env python3
"""
Bug #0033 — onnxruntime (ORT_ENABLE_ALL) diverges from pytorch_eager.

Patterns  : [['constant', 'identity_chain_two'], ['fusion', 'conv_tanh_head'], ['layout', 'slice_pad_concat'], ['broadcast', 'floor_unary'], ['broadcast', 'ceil_unary'], ['attention', 'matmul_4d_batch']]
S_diff    : 1.000000   (ORT_opt vs pytorch_eager)
Delta_opt : 0.000066   (ORT_opt vs ORT_noopt)
Tolerance : 0.001

    python unique_0033.py
Exit 0 = reproduced, 1 = not reproduced.
"""
import base64, sys
import numpy as np
import onnx
import onnxruntime as ort

TOLERANCE = 0.001

_ONNX_B64 = (
    "CAg64i8KIgoLbW9kZWxfaW5wdXQSCW4wX2ljMl9pMSIISWRlbnRpdHkKIQoJbjBfaWMyX2kxEgpu"
    "MF9pYzJfb3V0IghJZGVudGl0eQphCgpuMF9pYzJfb3V0CgpuMTAwX2N0aF93CgpuMTAwX2N0aF9i"
    "EgtuMTAwX2N0aF9jdiIEQ29udioVCgxrZXJuZWxfc2hhcGVAAUABoAEHKhEKBHBhZHNAAEAAQABA"
    "AKABBwohCgtuMTAwX2N0aF9jdhIMbjEwMF9jdGhfb3V0IgRUYW5oCj4KDG4xMDBfY3RoX291dAoM"
    "bjIwMF9zcGNfc2xzCgxuMjAwX3NwY19zbGUSC24yMDBfc3BjX3NsIgVTbGljZQpDCgtuMjAwX3Nw"
    "Y19zbAoNbjIwMF9zcGNfcGFkcxILbjIwMF9zcGNfcGQiA1BhZCoTCgRtb2RlIghjb25zdGFudKAB"
    "Awo+CgxuMTAwX2N0aF9vdXQKC24yMDBfc3BjX3BkEgxuMjAwX3NwY19vdXQiBkNvbmNhdCoLCgRh"
    "eGlzGAGgAQIKIgoMbjIwMF9zcGNfb3V0EgtuMzAwX2Zsb29fYSIFRmxvb3IKLwoLbjMwMF9mbG9v"
    "X2EKDG4yMDBfc3BjX291dBINbjMwMF9mbG9vX291dCIDTXVsCiIKDW4zMDBfZmxvb19vdXQSC240"
    "MDBfY2VpbF9hIgRDZWlsCjAKC240MDBfY2VpbF9hCg1uMzAwX2Zsb29fb3V0Eg1uNDAwX2NlaWxf"
    "b3V0IgNNdWwKMwoNbjQwMF9jZWlsX291dAoLbjUwMF9tbTRkX3cSDW41MDBfbW00ZF9vdXQiBk1h"
    "dE11bBILdHJpb25fZ3JhcGgqmQYIQAgDCAEIARABQgpuMTAwX2N0aF93SoAGB7WLPxQYhL+0lqm6"
    "+i61P6os5D6vaAK/XtkmP4BjzzzaPXE+mjgevhdFGD5LMnm/z6wqvxMpFD4UQhm/JygiP0Opdb/d"
    "Y5C/6OyrvrXekz92pZw+CsDNPtLDhT8mM9+/41gzvhIZYT/Yx+w882NRv5ae1T8jxQ0/OQiVP4se"
    "pz6sShe+GuMiPwWNmj9FYiC/WOGZPzgoPT/rooY+mRCSv2Stu776LbU9agorv01Cjr/JcQa/ECoA"
    "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"
    "GAtAvn2Xij0rNtw+Rj71PYBsEr44HZI+S8m2vZFWvz7eZ4e+kFWIvfeBtj0XAxG+VYWkvUqQU72C"
    "5oK+FmLwvRKPST73lgA7rOnRPdjw9r5f48G9YVb0PSnITj6vSXq+f9IJPtEqEb9QrwO+DlfBPZyQ"
    "qD0Kvl+8RZAEvidWYD560Va+3ZUOvp6LMT4o/Qe+87SLPpxLar5A5Qy+8nVNO90wED7OhSI+iGZ7"
    "PkF7pj3fkYK+HxlrPhVbhD70rhs+Sj7NPERBij5UfoM+bSeuPim4GT63M2c+ylL7vEhcmj2CV2w9"
    "h7LOvSlQPj7voMy+yQzDvW2rBT6MdgA/PKi/vbcrB74YQzu+qbawvfYAgT4Kab6+jEHgPN0Blj7b"
    "alu9BZCzvZreDb6d+ko+dZ/KveNel71JwuO9vAcFPjUoab6ZvEm+xEVovrxHxjxB6ha+uAs5Pkuk"
    "YT4U/6y+Ey/yvlNn5j5of7G+ArUKvgqSnT2Wz/o+S3yfvogSVj5/szM8ypkRvp8JPj3eKC4+SrOu"
    "vvpu2ryyvQY+KzPlvhihiz5MeSK+ntDuvBUgn75Su3o9O7JOuB1Ta77Snos+VGK/PjWYkb4l32c6"
    "An+LvmV4Cj707a0+jnyzPrD+qL76Xzw+R/afO2tucj1S8rM8fU5qvj+ZpL4JDoS9uZarvqt/Bz63"
    "DTa/4DXYvj2ZGb67VDM+8mdJvZkOlD63t0+9vPiRPUnykz71oSW+MJIfvpiaJ76ty0++kXArvgkG"
    "ID3+Qrk+TSHNPnwJKD0TQHO+faYAv++5LT4dRYs+N9WzvTyGZT7j/Um+c2JmPl1IRT6jhLs+uQul"
    "PUjzHz6K/Vq+phUovnW+4D6sfgs826WRPt7ktr4cq3G95qJhvtQfCT5zlRw+/CB6Pt8vNT32ahs+"
    "waBovkjv0D2SeOy9LYy7vlVyhj48Qxk9zyAIPtyjjL3liDC8B1phvtLPIT4mOoQ+SoQwPhllYDvT"
    "qK4+EeWPPnQ83j3Y8tS+y5AdPmrHdb4yuiU+Gj+avumBi711Sr4+loxqPtcbuT4scLm7b0DuPtVG"
    "MD6HEpm+U3IYv4EYHjsfsoa+TUg6v+OJaD4heaa+R5qYvhyJxT5+JcE+fB6evklF3TyDeca7PIZm"
    "PgeZQL7yL5y92+TiPem6Kr52D5s+n2v8Pg6G4rsvseA+P7DhPl+mm7zdYr0+4pB9PoxmwD5aJQoL"
    "bW9kZWxfaW5wdXQSFgoUCAESEAoCCAEKAggDCgIIIAoCCCBiKAoNbjUwMF9tbTRkX291dBIXChUI"
    "ARIRCgIIAQoDCIABCgIIIAoCCCBCBAoAEBE="
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
