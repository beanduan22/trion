#!/usr/bin/env python3
"""
Bug v2-0038 — OnnxRuntime wrong output vs ONNX spec reference.
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT incorrectly eliminates sub_self (x-x→0) and mul_by_one (x*1→x) in sequence — the combined elimination removes computation that was shaping the tensor layout for subsequent relu.
Report to : https://github.com/microsoft/onnxruntime/issues
           (label: bug, regression, Resize / TopK / CumSum etc.)

Oracle: ORT_ENABLE_ALL output vs pytorch_eager (onnx2torch) reference.
        ORT_DISABLE_ALL gives the same wrong answer — this is a fundamental
        implementation divergence, not an optimization regression.

The ONNX model is embedded in unique_2657.py (the reference repro).
Run: python v2_0038_onnxruntime.py  →  exit 0 = BUG REPRODUCED
"""
import base64, sys
import numpy as np
import onnx
import onnxruntime as ort

TOLERANCE = 0.01

_ONNX_B64 = (
    "CAg66xMKHgoLbW9kZWxfaW5wdXQSCW4wX3Jhcl9yMSIEUmVsdQooCgluMF9yYXJfcjEKC21vZGVs"
    "X2lucHV0EgluMF9yYXJfYWQiA0FkZAodCgluMF9yYXJfYWQSCm4wX3Jhcl9vdXQiBFJlbHUKLAoK"
    "bjBfcmFyX291dAoMbjEwMF9tYm9fb25lEgtuMTAwX21ib19tMSIDTXVsCi0KC24xMDBfbWJvX20x"
    "CgxuMTAwX21ib19vbmUSC24xMDBfbWJvX20yIgNNdWwKLgoLbjEwMF9tYm9fbTIKDG4xMDBfbWJv"
    "X29uZRIMbjEwMF9tYm9fb3V0IgNNdWwKLQoMbjEwMF9tYm9fb3V0CgxuMTAwX21ib19vdXQSCm4y"
    "MDBfc3N6X3oiA1N1YgotCgxuMTAwX21ib19vdXQKCm4yMDBfc3N6X3oSDG4yMDBfc3N6X291dCID"
    "QWRkCm8KDG4yMDBfc3N6X291dAoJbjMwMF90a19rEgxuMzAwX3RrX3ZhbHMSC24zMDBfdGtfaWR4"
    "IgRUb3BLKhQKBGF4aXMY////////////AaABAioOCgdsYXJnZXN0GAGgAQIqDQoGc29ydGVkGAGg"
    "AQIKLgoMbjMwMF90a192YWxzCgtuMzAwX3RrX3JlcBILbjMwMF90a19vdXQiBFRpbGUKQQoObjQw"
    "MF9nbG5fZW1iZWQKDG40MDBfZ2xuX2lkeBIMbjQwMF9nbG5fZ3RoIgZHYXRoZXIqCwoEYXhpcxgA"
    "oAECCjAKDG40MDBfZ2xuX2d0aAoLbjMwMF90a19vdXQSDm40MDBfZ2xuX2FkZGVkIgNBZGQKcwoO"
    "bjQwMF9nbG5fYWRkZWQKCm40MDBfZ2xuX2cKCm40MDBfZ2xuX2ISDG40MDBfZ2xuX291dCISTGF5"
    "ZXJOb3JtYWxpemF0aW9uKhQKBGF4aXMY////////////AaABAioRCgdlcHNpbG9uFazFJzegAQES"
    "C3RyaW9uX2dyYXBoKhgIARABQgxuMTAwX21ib19vbmVKBAAAgD8qGQgBEAdCCW4zMDBfdGtfa0oI"
    "AQAAAAAAAAAqIwgCEAdCC24zMDBfdGtfcmVwShABAAAAAAAAAEAAAAAAAAAAKpkICAQIQBABQg5u"
    "NDAwX2dsbl9lbWJlZEqACLM5j7yleie90q4JPVv2MjuCccu8Uf0BvGfDJjx5Nf68wQThvLe3NDwA"
    "gV+87kS5PKt4+7ttqcm7oqJAPCa48jxSkZo88it0PIkQvLxYOkK8tC7DPLrRC71YM3S7uewKPB6x"
    "jLxyVts7U5rju7MmqjxYioI7lZZevMLLKLyiD9e80uTuu5HGAb0igTo8HgF6vAJ9jzwXKb66AhV0"
    "vFHojbitBJc8vUyFPOykarxPMrq6rDnbu8VFfDyzEdG7qjOXPLL1oDs9+1K8tvmCvPLMcjuc3fy8"
    "qU7DvNyqNj18LTY8PRTWu1HWBjyPDcM7YW3UOxOpqbtPeFi8lCwCu+1FpTystxe8UzvOPNHyFj3Y"
    "TXm9SSW4O2+6cjzEEjU8irQmO4/TSTyU/4W9GRqguzITxTyak4A855ePumseLDwmKPA8gfD/uSQ4"
    "lLsA0vw8LH52POF/aLyqW4u8glMkvbtNlzyya/G8IK3WOwrHGTw0qBA8iQdAvUQOTzxyfQc8InI0"
    "PLj5rT1ExoI8lloNvXDEtLy28gE8YA/rvJ0xort1A/67G9mUvE5BKrx/o4a8ORbzO8d/TjpvwQG9"
    "9GEuvPEpijzic5C8L77ZvNvI9bxJPge7m6PlvIFtOjwGowi9wZyIvFuLDb0G1qO70NwIvWGDUzyA"
    "tf269jYCPQqIDbwQkco8CehRvdernjwELDg8Z7ujvEzG77yVtPM8BsxYvMIpczwkvEu8Va6CvGcM"
    "rTugWsG7S7q3PBDSY7zn1ke4JwN0PPazgjzTKn48781yu3TS6TkWHAQ8S8JmOoD+PLxSqYU6RBkd"
    "PKNS/jx/LBw8x7eAO8uWtrhwQMg7W/0RPEGuzDxBgba8GWEPPPNg4rvFqmA8/+eLvKBB2bxll6C8"
    "RhYivDZhCz2Tok88Jwdku68QYDujLfA8kAGuu9CQ/jvuJ5u7kFROPP5WLTzkuNq7AXK7vAE3NLwS"
    "MSo8Eo2pPM+AAL3qVkq8/Q8iO86w3Lp5BhI854AEvUfYULzShn88E3q1O0VYrjwtbLa7mbR3u6q9"
    "wbys7t4854X5vJUVTDztsRk9Be2NPJXoTL3Cn488QwgRvRbxCrxKqts7i0OUvAd8BTyskhM8tDjb"
    "O6nACT3Z1M+7W4R3vKkfsDs+MXC8Hm31PFPWhbyGG8a8LYIMu7wuST3AQHc88rcsPGJuBDui12g7"
    "R4nuvOZeAbwJHPO8GyaVvNBEvDwe7sE7eZYlPHutLLvYEyo6qpe8u7+ZhLzr99w7bOh0PJIRdbtk"
    "sxi9558VvfxEurwnizo8zPIKvVgK2Twpapm8XAFqvL1CpTwrrAQ99KvtvI46/bv5BoO8fsvAPL1z"
    "27zBBy69JuoqvR2aID0qHAgBEAdCDG40MDBfZ2xuX2lkeEoIAAAAAAAAAAAqkwIIQBABQgpuNDAw"
    "X2dsbl9nSoACAACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
    "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
    "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
    "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPyqTAghAEAFCCm40MDBfZ2xuX2JK"
    "gAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWh0KC21vZGVsX2lucHV0Eg4KDAgBEggKAggB"
    "CgIIQGIeCgxuNDAwX2dsbl9vdXQSDgoMCAESCAoCCAEKAghAQgQKABAR"
)
_INPUTS = {
    'model_input': (
    "pn3ePT+awz8kMCM9OrCJPz+hEMAhh4g/P6vqvoy1jcDEFEVAvcwZwOULOsANg5i+7TjSPypPckAL"
    "zus/vm2DQA0yBD71SV3AT/oNP2q61z2xgfE/bdHDPoMDrEAyFTXAh3Eiv6GN/b7VH4S9+XAMvvi8"
    "IMCE5d2/I+d7v0U9JEAP7sq/JhM0wIllDsAlrT0/vTkXQKutgD6QTiBAKJqAQDu7E0DtWeo9UURt"
    "P1nkRT+YNRK/3NDzv//Q0b9lEjW/t49RwOCMOj9Km3I+Zb7EPsZ5Q8C73UhAi20RwCfqmj95mv2/"
    "wfA9v7FyfsDPQSPAzy9/QPYyFkDiFivA2qwTvw==",
        [1, 64],
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
