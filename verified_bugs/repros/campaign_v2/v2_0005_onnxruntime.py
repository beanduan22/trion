#!/usr/bin/env python3
"""
Bug v2-0005 — OnnxRuntime wrong output vs ONNX spec reference.
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT TopK(k=1, axis=-1) returns incorrect indices when input values are all identical (from x-x=0 pattern) — optimizer's constant folding of x-x→0 does not preserve the expected tie-breaking order.
Report to : https://github.com/microsoft/onnxruntime/issues
           (label: bug, regression, Resize / TopK / CumSum etc.)

Oracle: ORT_ENABLE_ALL output vs pytorch_eager (onnx2torch) reference.
        ORT_DISABLE_ALL gives the same wrong answer — this is a fundamental
        implementation divergence, not an optimization regression.

The ONNX model is embedded in unique_2624.py (the reference repro).
Run: python v2_0005_onnxruntime.py  →  exit 0 = BUG REPRODUCED
"""
import base64, sys
import numpy as np
import onnx
import onnxruntime as ort

TOLERANCE = 0.01

_ONNX_B64 = (
    "CAg62BwKaAoLbW9kZWxfaW5wdXQKB24wX3RrX2sSCm4wX3RrX3ZhbHMSCW4wX3RrX2lkeCIEVG9w"
    "SyoUCgRheGlzGP///////////wGgAQIqDgoHbGFyZ2VzdBgBoAECKg0KBnNvcnRlZBgBoAECCigK"
    "Cm4wX3RrX3ZhbHMKCW4wX3RrX3JlcBIJbjBfdGtfb3V0IgRUaWxlCigKCW4wX3RrX291dAoJbjBf"
    "dGtfb3V0EgtuMTAwX3RhZF9hMSIDQWRkCioKC24xMDBfdGFkX2ExCgluMF90a19vdXQSC24xMDBf"
    "dGFkX2EyIgNBZGQKKwoLbjEwMF90YWRfYTIKCW4wX3RrX291dBIMbjEwMF90YWRfb3V0IgNBZGQK"
    "TgoMbjEwMF90YWRfb3V0Eg1uMjAwX3JlZHVfcmVkIghSZWR1Y2VMMioUCgRheGVzQP//////////"
    "/wGgAQcqDwoIa2VlcGRpbXMYAaABAgowCgxuMTAwX3RhZF9vdXQKDm4yMDBfcmVkdV96ZXJvEgtu"
    "MjAwX3JlZHVfeiIDTXVsCjAKC24yMDBfcmVkdV96Cg1uMjAwX3JlZHVfcmVkEg1uMjAwX3JlZHVf"
    "b3V0IgNBZGQKJgoNbjIwMF9yZWR1X291dBILbjMwMF9pYzJfaTEiCElkZW50aXR5CiUKC24zMDBf"
    "aWMyX2kxEgxuMzAwX2ljMl9vdXQiCElkZW50aXR5CnMKDG4zMDBfaWMyX291dAoLbjQwMF9kbG5f"
    "ZzEKC240MDBfZGxuX2IxEgxuNDAwX2Rsbl9sbjEiEkxheWVyTm9ybWFsaXphdGlvbioUCgRheGlz"
    "GP///////////wGgAQIqEQoHZXBzaWxvbhWsxSc3oAEBCjAKDG40MDBfZGxuX2xuMQoObjQwMF9k"
    "bG5fc2NhbGUSC240MDBfZGxuX3NjIgNNdWwKcgoLbjQwMF9kbG5fc2MKC240MDBfZGxuX2cyCgtu"
    "NDAwX2Rsbl9iMhIMbjQwMF9kbG5fb3V0IhJMYXllck5vcm1hbGl6YXRpb24qFAoEYXhpcxj/////"
    "//////8BoAECKhEKB2Vwc2lsb24VrMUnN6ABARILdHJpb25fZ3JhcGgqFwgBEAdCB24wX3RrX2tK"
    "CAEAAAAAAAAAKiEIAhAHQgluMF90a19yZXBKEAEAAAAAAAAAgAAAAAAAAAAqGggBEAFCDm4yMDBf"
    "cmVkdV96ZXJvSgQAAAAAKpUECIABEAFCC240MDBfZGxuX2cxSoAEAACAPwAAgD8AAIA/AACAPwAA"
    "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
    "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
    "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
    "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
    "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
    "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
    "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8qlQQIgAEQAUILbjQwMF9kbG5f"
    "YjFKgAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAACqVBAiAARABQgtuNDAwX2Rsbl9nMkqABAAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
    "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
    "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
    "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
    "gD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACA"
    "PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
    "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/KpUECIABEAFCC240MDBfZGxuX2IySoAEAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqmgQIAQiA"
    "ARABQg5uNDAwX2Rsbl9zY2FsZUqABM3MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyM"
    "P83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/"
    "zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/N"
    "zIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83M"
    "jD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyM"
    "P83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/"
    "zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/N"
    "zIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83M"
    "jD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyMP83MjD/NzIw/zcyM"
    "P83MjD/NzIw/zcyMP83MjD/NzIw/Wh4KC21vZGVsX2lucHV0Eg8KDQgBEgkKAggBCgMIgAFiHwoM"
    "bjQwMF9kbG5fb3V0Eg8KDQgBEgkKAggBCgMIgAFCBAoAEBE="
)
_INPUTS = {
    'model_input': (
    "thzhv5ljCj8qf6k/7f7GP7dqM7+b7ni9PN0AwFSYgT3s6Ye/uvxjvla0dz/FrWu/KDpJPocTGb+O"
    "6sK+z5UzPoQg5r95m1S/e3zRv/QAlD6QOTg+aGiHv799/z47AmC/sgy+P0XrpT+PioK/mXaHv4cQ"
    "gD4bw+y/C5yZv+cBZD+DLM8/7D0kv1Km3j7sewW/IWyxPr8f9j6cGg29ZtiyPneRGz8v7jW/x9Gk"
    "P2LBNT8bGPu+KlJBP9bDyr9DM5i/6nSFv0grrr8KayU/03+xvqu5EL/7Cxk/tSduvb5kj79PZLQ+"
    "PMriP5Rld7/eFz2/MGiKvx52uj5cJaE/iNUqP155Lr/7NOu9yL8cQPCg/r7IcM6+bXB/v9pZ1b8G"
    "On4//ccvP4CgJz+vWs4+YI4kv8PGGD+GWiy/Pv65vmLZkb3t7I+/5hFdvuajFT8aF8e6wuKhPmHc"
    "ZL/6PR6+lsdBPghm6j3aXZA/JPazPw7K/r8lfqO9HgGcvj0x0D5sjOG9qvqIP7TU3L5WkS+/7OuC"
    "v24hiD5inAbAJsbIPtZ6dD8vVZw9KXQCPIzGjD+bilo/V9PZvfx2KD6/wp2/3IvtPqWZw775RbM/"
    "w6O1PzgVT7+Nz629RvZmPqDVOb8qj5K/CD0VO3tO7L6zhaS+F3TrP0yHfb/9oec+PdWeP5YGfL0=",
        [1, 128],
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
