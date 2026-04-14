"""
OpenVINO Bug: MatMul GPU plugin gives wrong output when inner dim > 2048.

Source : https://github.com/openvinotoolkit/openvino/issues/22613
Affects: OpenVINO 2023.0–2023.3, GPU plugin; fixed in 2024.x
Root cause: GPU kernel for MatMul used a tile size of 2048; inputs
            exceeding that size caused the last tile to be skipped,
            so the accumulator holds a partial (off-by-one-tile) sum.

Shape: [1, 1, size] × [1, size, 1] → [1, 1, 1]   (dot product of ones = size)

Expected: result == size  for all size values
Actual  : result == 2048  when size > 2048  (GPU only)
"""
import numpy as np

try:
    import openvino as ov
    HAS_OV = True
except ImportError:
    HAS_OV = False


def make_matmul_model(size):
    import openvino.runtime as ov_rt
    p1 = ov_rt.opset10.parameter([1, 1, size], ov_rt.Type.f32, name="A")
    p2 = ov_rt.opset10.parameter([1, size, 1], ov_rt.Type.f32, name="B")
    mm = ov_rt.opset10.matmul(p1, p2, transpose_a=False, transpose_b=False)
    return ov_rt.Model([mm], [p1, p2], "matmul_model")


def run(size, device="CPU"):
    core  = ov.Core()
    model = make_matmul_model(size)
    cmod  = core.compile_model(model, device)
    A     = np.ones((1, 1, size), dtype=np.float32)
    B     = np.ones((1, size, 1), dtype=np.float32)
    result = cmod([A, B])[0].flat[0]
    return float(result)


if not HAS_OV:
    print("OpenVINO not installed — showing expected vs buggy analytically.")
    for size in [2048, 2049, 4096]:
        expected = float(size)
        buggy    = 2048.0 if size > 2048 else float(size)
        PASS_    = abs(expected - buggy) < 1.0
        print(f"size={size:5d}  expected={expected:.0f}  gpu_result={buggy:.0f}  PASS={PASS_}")
else:
    results = {}
    for size in [2048, 2049, 4096]:
        cpu_res = run(size, "CPU")
        results[size] = cpu_res
        print(f"size={size:5d}  CPU={cpu_res:.1f}  expected={size:.1f}")

    # Try GPU if available
    devices = ov.Core().available_devices
    if "GPU" in devices:
        for size in [2048, 2049, 4096]:
            gpu_res = run(size, "GPU")
            PASS_ = abs(gpu_res - size) < 1.0
            print(f"size={size:5d}  GPU={gpu_res:.1f}  expected={size:.1f}  PASS={PASS_}")
    else:
        print("GPU device not available — CPU baseline only.")
        PASS = all(abs(results[s] - s) < 1.0 for s in results)
        print(f"PASS={PASS}  (GPU bug requires GPU device to reproduce)")
