#!/usr/bin/env python3
"""
Bug v2-0047 — OnnxRuntime wrong output vs ONNX spec reference.
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Root cause: ORT L2 norm (ReduceSum+Sqrt) gives wrong result after mul+add chain — optimizer fuses the arithmetic into the reduction kernel using different float ordering.
Report to : https://github.com/microsoft/onnxruntime/issues
           (label: bug, regression, Resize / TopK / CumSum etc.)

Oracle: ORT_ENABLE_ALL output vs pytorch_eager (onnx2torch) reference.
        ORT_DISABLE_ALL gives the same wrong answer — this is a fundamental
        implementation divergence, not an optimization regression.

The ONNX model is embedded in unique_2666.py (the reference repro).
Run: python v2_0047_onnxruntime.py  →  exit 0 = BUG REPRODUCED
"""
import base64, sys
import numpy as np
import onnx
import onnxruntime as ort

TOLERANCE = 0.01

_ONNX_B64 = (
    "CAg6hyMKKwoLbW9kZWxfaW5wdXQKC21vZGVsX2lucHV0EgpuMF9sMm5tX3NxIgNNdWwKQAoKbjBf"
    "bDJubV9zcQoKbjBfbDJubV9heBIKbjBfbDJubV9zbSIJUmVkdWNlU3VtKg8KCGtlZXBkaW1zGAGg"
    "AQIKKgoKbjBfbDJubV9zbQoLbjBfbDJubV9lcHMSCm4wX2wybm1fYWQiA0FkZAoeCgpuMF9sMm5t"
    "X2FkEgpuMF9sMm5tX3NyIgRTcXJ0CisKC21vZGVsX2lucHV0CgpuMF9sMm5tX3NyEgtuMF9sMm5t"
    "X291dCIDRGl2CiwKC24wX2wybm1fb3V0CgpuMTAwX2RiY19jEgxuMTAwX2RiY19kaXYiA0Rpdgot"
    "CgtuMF9sMm5tX291dAoLbjEwMF9kYmNfaWMSDG4xMDBfZGJjX211bCIDTXVsCi8KDG4xMDBfZGJj"
    "X2RpdgoMbjEwMF9kYmNfbXVsEgxuMTAwX2RiY19zdW0iA0FkZAovCgxuMTAwX2RiY19zdW0KDG4x"
    "MDBfZGJjX3R3bxIMbjEwMF9kYmNfb3V0IgNEaXYKLAoMbjEwMF9kYmNfb3V0CgpuMjAwX21hbV9h"
    "EgtuMjAwX21hbV94MSIDTXVsCisKC24yMDBfbWFtX3gxCgpuMjAwX21hbV9iEgtuMjAwX21hbV94"
    "MiIDQWRkCiwKC24yMDBfbWFtX3gyCgpuMjAwX21hbV9jEgxuMjAwX21hbV9vdXQiA011bApMCgxu"
    "MjAwX21hbV9vdXQKDG4zMDBfbWluX19jMAoMbjMwMF9taW5fX2MxCgxuMzAwX21pbl9fYzISDW4z"
    "MDBfbWluX19vdXQiA01pbgpOCg1uMzAwX21pbl9fb3V0EgpuNDAwX21sbl9tIgpSZWR1Y2VNZWFu"
    "KhQKBGF4ZXNA////////////AaABByoPCghrZWVwZGltcxgBoAECCi4KDW4zMDBfbWluX19vdXQK"
    "Cm40MDBfbWxuX20SDG40MDBfbWxuX3N1YiIDU3ViCi4KDG40MDBfbWxuX3N1YgoMbjQwMF9tbG5f"
    "c3ViEgtuNDAwX21sbl9zcSIDTXVsCk4KC240MDBfbWxuX3NxEgxuNDAwX21sbl92YXIiClJlZHVj"
    "ZU1lYW4qFAoEYXhlc0D///////////8BoAEHKg8KCGtlZXBkaW1zGAGgAQIKLwoMbjQwMF9tbG5f"
    "dmFyCgxuNDAwX21sbl9lcHMSDG40MDBfbWxuX2FkZCIDQWRkCiMKDG40MDBfbWxuX2FkZBINbjQw"
    "MF9tbG5fc3FydCIEU3FydAowCgxuNDAwX21sbl9zdWIKDW40MDBfbWxuX3NxcnQSDG40MDBfbWxu"
    "X291dCIDRGl2Egt0cmlvbl9ncmFwaCoXCAEQAUILbjBfbDJubV9lcHNKBHfMKzIqGggBEAdCCm4w"
    "X2wybm1fYXhKCP//////////KhYIARABQgpuMTAwX2RiY19jSgQAAKhAKhcIARABQgtuMTAwX2Ri"
    "Y19pY0oEMQxDPioYCAEQAUIMbjEwMF9kYmNfdHdvSgQAAABAKhYIARABQgpuMjAwX21hbV9hSgSa"
    "mZk/KhYIARABQgpuMjAwX21hbV9iSgSamZk+KhYIARABQgpuMjAwX21hbV9jSgQAAMA/KpgICAEI"
    "gAIQAUIMbjMwMF9taW5fX2MwSoAIzczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9"
    "zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3N"
    "zMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3M"
    "zD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczM"
    "Pc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9"
    "zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3N"
    "zMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3M"
    "zD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczM"
    "Pc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9"
    "zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3N"
    "zMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3M"
    "zD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczM"
    "Pc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9"
    "zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3N"
    "zMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3M"
    "zD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczM"
    "Pc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9"
    "zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3NzMw9zczMPc3MzD3N"
    "zMw9zczMPc3MzD3NzMw9zczMPSqYCAgBCIACEAFCDG4zMDBfbWluX19jMUqACM3MTD7NzEw+zcxM"
    "Ps3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+"
    "zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7N"
    "zEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3M"
    "TD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxM"
    "Ps3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+"
    "zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7N"
    "zEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3M"
    "TD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxM"
    "Ps3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+"
    "zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7N"
    "zEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3M"
    "TD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxM"
    "Ps3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+"
    "zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7N"
    "zEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3M"
    "TD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxM"
    "Ps3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+"
    "zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD7NzEw+zcxMPs3MTD4qmAgIAQiAAhABQgxu"
    "MzAwX21pbl9fYzJKgAiamZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZ"
    "PpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+"
    "mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6a"
    "mZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZ"
    "mT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZ"
    "PpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+"
    "mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6a"
    "mZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZ"
    "mT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZ"
    "PpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+"
    "mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6a"
    "mZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZ"
    "mT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZ"
    "PpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+"
    "mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6a"
    "mZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZ"
    "mT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZ"
    "PpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+mpmZPpqZmT6amZk+"
    "mpmZPpqZmT6amZk+KhgIARABQgxuNDAwX21sbl9lcHNKBKzFJzdaHgoLbW9kZWxfaW5wdXQSDwoN"
    "CAESCQoCCAEKAwiAAmIfCgxuNDAwX21sbl9vdXQSDwoNCAESCQoCCAEKAwiAAkIECgAQEQ=="
)
_INPUTS = {
    'model_input': (
    "gLHLvs2lVb+mRe6/oRq5POFrkL9TokHAKL/YuoM2Oz+A8xa/HWYIvqo1wT12Epy+ukxcP9PQ2b7c"
    "7wTA2z5ivxvw6b7Vuk2/s36uvxlKBsAtJ8W9DTsiP6GiqT/KABm9s/hJPzto/zwURxE/W1RCPim7"
    "jL8FeME/PUG1vmO+mb7n1Ae/I8h6Pi3G275wLNQ/QCi2P81xKL8h27w+OOjUvxtWi7/Gze8+FArx"
    "vdisAD/EdVS/O+AvPEw9Ub/pyhA/Q90pv7QMEcB1sbc+s3CpP1L/frsBIru/I/J3vuN52T8oI1W/"
    "LywovyZThL9Jjn8/1JNqv/RCMb+2XV+/VttNP8MIWD4ri7G/eMqIvygDRsAjjAo/CyeLP+P5wT1a"
    "j5c/mWdIvxJDDb+FDNC/gCXIv7aLBr9U+Lm/IGOAvwKY474Wt3U/O4ydu7xRkr071eO+gpSzP8PJ"
    "Xj/F0jk/N2i0vgi68z7kt7w/jGeNPqhC4D9h9HY/G7zPPgc8TT9roXs9Ne6jPvgNKT+GdWq/uAYn"
    "P3RTRr+rUOK/Yk2fPmdixj7UiDTAEOIFwGgQhD5Oo6W/Msq2PsumJj9bdhTAFvgIPCrQw76wxEI/"
    "sfisvwlEdL++BHq/UGB4v5CyI8D9SgA/1CutPw5vrj6SLh4/r1PBvs5EpT/qPoM+7JfuvqP2BT4N"
    "Zuo+mn0tPqXjBD/MhPc+wXuwvujlHD+fzKQ+Ycolv+OOID/hhmU8Cj2jvxVrjj+NUzo/1Mgmv/th"
    "mz6js62/WStwvZi1zj9nzNq9Gu1TP3H30j/E4+m+fTbVPbOFl7/4pfE+74mJPgN1oz+xYom/PG1V"
    "v0+uoz/liQnA9CmGvzgfkb5bxoS/ohdWP5c6qD9h0Ne+c/xlvYDUqD/H7rA//7ZOP2sfrT9bytC+"
    "86EFwNImBT0xy+q/iX6Mv0cOEr/qQVY/NxCyPnqcuL4pYeE/H72aP5HxDj8eBks+u1Aevag6cD5V"
    "+gY+kWtuv0tuIj+2NiU+7u9ZP7J+SL+T9Qq/e34+P2MqZb7FR+e9OtTAvzTqtD7oTyQ/VELePyVs"
    "tj9h/Yu/Hvh+Pjwycr7+Pve+cIA5P6Xy4z4eolM/TTf6P1cCmj5KfYu/EMhaP6WWP79ZeX8/8c+e"
    "PWOOsj6e13A+eYyOPg1hAD7yY5a/4RHQv5mRXz+53f4+s8KSvD9Iqz3ehmg/16u8P7ZhB7+9Maa+"
    "jUu/vrwTBr9b74Y/5D9UP79LLT+iXIq/jl0GQGlyxr/+gHY+3pggP6rATz/Z7j6/opAJP7OqY75Y"
    "YuK/OxgCv0Xngj6Tsh6/6N/DP8+r7j6bgU4/2NtAvxhmjr5HoKg/7+2vP5gKkD4K6Ja++tAPPw==",
        [1, 256],
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
