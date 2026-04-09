#!/usr/bin/env python3
"""
Bug #0018 — onnxruntime (ORT_ENABLE_ALL) diverges from pytorch_eager.

Patterns  : [['constant', 'neg_abs_identity'], ['fusion', 'conv_clip_conv_clip_mbv2'], ['layout', 'resize_linear_asymmetric'], ['layout', 'gather_channel_permute'], ['layout', 'unsqueeze_expand_mul'], ['normalization', 'layernorm_dropout_identity']]
S_diff    : 1.000000   (ORT_opt vs pytorch_eager)
Delta_opt : 0.000000   (ORT_opt vs ORT_noopt)
Tolerance : 0.001

    python unique_0018.py
Exit 0 = reproduced, 1 = not reproduced.
"""
import base64, sys
import numpy as np
import onnx
import onnxruntime as ort

TOLERANCE = 0.001

_ONNX_B64 = (
    "CAg6kRgKHQoLbW9kZWxfaW5wdXQSCW4wX25haV9uZyIDTmVnChwKCW4wX25haV9uZxIKbjBfbmFp"
    "X291dCIDQWJzCmcKCm4wX25haV9vdXQKDG4xMDBfY2NjY193MQoMbjEwMF9jY2NjX2IxEg1uMTAw"
    "X2NjY2NfY3YxIgRDb252KhUKDGtlcm5lbF9zaGFwZUABQAGgAQcqEQoEcGFkc0AAQABAAEAAoAEH"
    "CkQKDW4xMDBfY2NjY19jdjEKDm4xMDBfY2NjY19jbWluCg5uMTAwX2NjY2NfY21heBINbjEwMF9j"
    "Y2NjX2NsMSIEQ2xpcApqCg1uMTAwX2NjY2NfY2wxCgxuMTAwX2NjY2NfdzIKDG4xMDBfY2NjY19i"
    "MhINbjEwMF9jY2NjX2N2MiIEQ29udioVCgxrZXJuZWxfc2hhcGVAAUABoAEHKhEKBHBhZHNAAEAA"
    "QABAAKABBwpECg1uMTAwX2NjY2NfY3YyCg5uMTAwX2NjY2NfY21pbgoObjEwMF9jY2NjX2NtYXgS"
    "DW4xMDBfY2NjY19vdXQiBENsaXAKiwEKDW4xMDBfY2NjY19vdXQKDW4yMDBfcmVzaV9yb2kKEG4y"
    "MDBfcmVzaV9zY2FsZXMSDW4yMDBfcmVzaV9vdXQiBlJlc2l6ZSovCh5jb29yZGluYXRlX3RyYW5z"
    "Zm9ybWF0aW9uX21vZGUiCmFzeW1tZXRyaWOgAQMqEQoEbW9kZSIGbGluZWFyoAEDCkAKDW4yMDBf"
    "cmVzaV9vdXQKDG4zMDBfZ2NwX2lkeBIMbjMwMF9nY3Bfb3V0IgZHYXRoZXIqCwoEYXhpcxgBoAEC"
    "CjcKDm40MDBfdWVtX3NjYWxlCg1uNDAwX3VlbV9heGVzEgtuNDAwX3VlbV91cyIJVW5zcXVlZXpl"
    "CjMKC240MDBfdWVtX3VzCg9uNDAwX3VlbV90c2hhcGUSC240MDBfdWVtX2V4IgZFeHBhbmQKLgoM"
    "bjMwMF9nY3Bfb3V0CgtuNDAwX3VlbV9leBIMbjQwMF91ZW1fb3V0IgNNdWwKcQoMbjQwMF91ZW1f"
    "b3V0CgtuNTAwX2xuZF9zYwoKbjUwMF9sbmRfYhILbjUwMF9sbmRfbG4iEkxheWVyTm9ybWFsaXph"
    "dGlvbioUCgRheGlzGP///////////wGgAQIqEQoHZXBzaWxvbhWsxSc3oAEBCjQKC241MDBfbG5k"
    "X2xuCg5uNTAwX2xuZF9yYXRpbxIMbjUwMF9sbmRfb3V0IgdEcm9wb3V0Egt0cmlvbl9ncmFwaCqb"
    "AwggCAMIAQgBEAFCDG4xMDBfY2NjY193MUqAA/61Aj8OiIQ9aFzLPviiMj7O+l8/RsYMv/HfyD5P"
    "7Ai+32sIP+IyGb4x3cY/b3OcPwTt/T5sr9m+gWtgPQDeG75hMdg+r5env4ZZNr+VhUk/7I5EP174"
    "N76RUmy+A9mgP4DOSb73qx8/B41PP1Beo7+JDAQ+8crWPweJLz/Pumy9y5sFPg2YMD5jIVI/o+Qh"
    "v6YVl7/8kx1AWTWyvr5tV756K1y/NFn1Puq7Jb913Uw/7yZ8vxTrTT/TNFy/+L6mvjyAGD/nu0a/"
    "bURwP4PVm7/p+YE/kQravmCVFr/Wihc/slxgv2HFQL0sLfE/dICBPwSxp79/pn2/1adBPxWOUr73"
    "bxjAGJVYvvzKqD6ZXwY/B3YfQAKz779MH7Q+YsdAPiKnYr4zhWw+8KJ5PZuFhb6rVzO/frOmviFo"
    "8j6pnhU/ITlRPTusCD/wse0+gYAMvxGiuD6VDbe+9el0vU4KGD9+mVQ+ZBEJP4QpVr+t3II+//LI"
    "PAxz0T6/OKa+xjcWPiqVAQggEAFCDG4xMDBfY2NjY19iMUqAAQAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKpsECAQI"
    "IAgBCAEQAUIMbjEwMF9jY2NjX3cySoAE7r1KPuLAib5/xpc822igPgAQFj8zmAw/7UdmPtaDwT0u"
    "a2M9YH67PUb4TT4i7YW+8dQSvibjqr27SXC+9NcePivp5b7hne8+gxLxvWI3xLzvNC8+jUbavQyp"
    "vz0oFsg99wLmPgQfsT6tKzU+R92xvoMStT5rtv09spHTPiI4Fz7s5q08s6nwvc/6XT7988m+e8FE"
    "vlTgoL6sQX2+QYSzPgzcoT5tyCS/XkOpPsYSd76+aBU8NEIxPjNlGb5GRpk8cSEDPz9zpjzQdIm+"
    "a+49PRhW8L2LpHc+kRjNPv3sq77V0Oi98F0VPndyRT7YaJK+b/ekPLBCYL3csLC+kBihve0ZJT5F"
    "19M+/xwSvy2/Bj9/bxW9BV+ovB1FJLuybK4+jGYuPuajiD4xyhU+aFaVvSUUXL4Xm6g+WTeVPs3/"
    "Lz41kRU/fYFEvkd4WD6L4ES+9vhCPWMm7rvgDVi9xFEJvhs3Yz1/K6u+bv9QPmfP5r5NMfm8OE95"
    "vXLECz9LEli+ZEaKPnFmhz74R5U+UMK6PvMhj7wmJaY+20lfvknN2D5BFxe+RjniPSlXmL0KVLM9"
    "6h2lvA6Zr74+5fM8RUpNvtSHHD+HLgG9m42WPhj1dL74a68+jXKEPp6mjb1RKsE+NZUEPl5qvr6i"
    "P36+04RJPaoZiD422mY+4rMvPvvSir4qJAgEEAFCDG4xMDBfY2NjY19iMkoQAAAAAAAAAAAAAAAA"
    "AAAAACoaCAEQAUIObjEwMF9jY2NjX2NtaW5KBAAAAAAqGggBEAFCDm4xMDBfY2NjY19jbWF4SgQA"
    "AMBAKhUIABABQg1uMjAwX3Jlc2lfcm9pSgAqKAgEEAFCEG4yMDBfcmVzaV9zY2FsZXNKEAAAgD8A"
    "AIA/AAAAQAAAAEAqNAgEEAdCDG4zMDBfZ2NwX2lkeEogAgAAAAAAAAABAAAAAAAAAAAAAAAAAAAA"
    "AwAAAAAAAAAqJggEEAFCDm40MDBfdWVtX3NjYWxlShAAAIA/AACAPwAAgD8AAIA/Ki0IAxAHQg1u"
    "NDAwX3VlbV9heGVzShgAAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAAqNwgEEAdCD240MDBfdWVtX3Rz"
    "aGFwZUogAQAAAAAAAAAEAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAAqlAIIQBABQgtuNTAwX2xuZF9z"
    "Y0qAAgAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
    "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
    "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
    "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8qkwIIQBABQgpuNTAwX2xuZF9iSoACAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACoYEAFCDm41MDBfbG5kX3JhdGlvSgQAAAAAWiUKC21v"
    "ZGVsX2lucHV0EhYKFAgBEhAKAggBCgIIAwoCCCAKAgggYiYKDG41MDBfbG5kX291dBIWChQIARIQ"
    "CgIIAQoCCAQKAghACgIIQEIECgAQEQ=="
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
