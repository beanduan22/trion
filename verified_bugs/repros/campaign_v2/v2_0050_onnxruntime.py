#!/usr/bin/env python3
"""
Bug v2-0050 — OnnxRuntime wrong output vs pytorch_eager reference.
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Source    : bug_000127 (model #127)
Patterns  : cumsum_last_axis, cast_roundtrip, sqrt_div_rms, topk_last_axis_k1, add_relu_sub
Root cause: [SUSPECT: all 6 compilers diverge] ORT miscomputes cumsum on last axis combined with int32 cast roundtrip, RMS-div, and TopK — SequentialCumSum may have off-by-one in last-axis prefix sum.
Report to : https://github.com/microsoft/onnxruntime/issues
NOTE: All 6 tested compilers diverge from pytorch_eager for this model.
        This MAY indicate a pytorch_eager (onnx2torch) reference bug rather than
        a compiler bug. Verify by comparing against the ONNX spec manually.
Oracle: ORT_ENABLE_ALL output vs pytorch_eager (onnx2torch).
ONNX model: /home/binduan/myspace/trion/trion_results/bug_000127.onnx
Run: python v2_0050_onnxruntime.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os
import numpy as np

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'trion_results', 'bug_000127.onnx')
INPUT = np.array([-0.10702719539403915, 2.364010810852051, -1.1202982664108276, -1.6814860105514526, -0.13097012042999268, 0.3444512188434601, 1.3036394119262695, -0.8519828915596008, 0.4179420471191406, -0.5610750913619995, -1.1727505922317505, 0.5024120807647705, 1.30174720287323, 0.08587296307086945, -0.26810258626937866, 0.03274695575237274, -0.765678346157074, 0.8303449749946594, -0.9576659798622131, 0.8363752961158752, 0.918917715549469, -2.055601119995117, -0.48182013630867004, 0.5085675120353699, 0.4987102448940277, 0.796829879283905, 0.5973130464553833, 0.5898944139480591, -1.7411117553710938, 1.092287540435791, 0.6855818629264832, 1.133784532546997, -0.45894739031791687, 0.3218260109424591, 0.2912411689758301, -1.2677414417266846, -0.5193027853965759, 0.6153186559677124, 0.3556828498840332, -1.2095260620117188, 0.08053980767726898, -0.7781769037246704, 0.0612487867474556, -0.5495544075965881, -0.9886614084243774, 0.05460387095808983, 1.3849830627441406, -1.2735732793807983, 0.8875815868377686, -0.9490528702735901, 0.9621091485023499, 0.6192609071731567, 1.3589999675750732, 1.7055045366287231, -0.831352710723877, -0.29752135276794434, 1.5866725444793701, -0.6068793535232544, 1.314582347869873, -0.22705760598182678, 0.5646356344223022, -0.4061303436756134, -1.5103166103363037, -1.8868697881698608], dtype=np.float32).reshape([1, 64])
TOLERANCE = 0.01

def _rel_l2(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.shape != b.shape: return float('inf')
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))

def _pytorch_eager():
    import onnx, onnx2torch, torch
    m = onnx2torch.convert(onnx.load(ONNX_PATH)).eval()
    with torch.no_grad():
        out = m(torch.from_numpy(INPUT))
    if isinstance(out, (list, tuple)): out = out[0]
    return out.float().numpy().ravel()

def main():
    if not os.path.exists(ONNX_PATH):
        print(f"ONNX model not found: {ONNX_PATH}")
        sys.exit(2)
    try:
        ref = _pytorch_eager()
    except Exception as e:
        print(f"[error] pytorch_eager reference failed: {e}")
        sys.exit(2)

    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(ONNX_PATH, opts, providers=["CPUExecutionProvider"])
        result = np.asarray(sess.run(None, {"model_input": INPUT})[0]).ravel()
    except Exception as e:
        print(f"[crash] onnxruntime: {e}")
        sys.exit(0)

    diff = _rel_l2(result, ref)
    print(f"Expected (pytorch_eager) at idx 0: {ref.ravel()[0]:.6f}")
    print(f"Got (OnnxRuntime) at idx 0: {result.ravel()[0]:.6f}")  # expected=1.545121, got=2.627084
    print(f"rel L2(ORT vs pytorch_eager) = {diff:.6e}")
    if diff > TOLERANCE:
        print("BUG REPRODUCED")
        sys.exit(0)
    print("not reproduced")
    sys.exit(1)

if __name__ == "__main__":
    main()
