#!/usr/bin/env python3
"""
Bug v2-0048 — OnnxRuntime wrong output vs pytorch_eager reference.
Compiler  : OnnxRuntime (ORT_ENABLE_ALL)
Source    : bug_000355 (model #355)
Patterns  : matmul_bias_gelu, crms_norm, mul_add_mul_chain, sqrt_reciprocal_mul, double_layernorm
Root cause: [SUSPECT: all 6 compilers diverge] ORT may miscompute matmul+bias+GELU under double layernorm with CRMSNorm — GELU tanh approximation vs exact erf may be the root cause.
Report to : https://github.com/microsoft/onnxruntime/issues
NOTE: All 6 tested compilers diverge from pytorch_eager for this model.
        This MAY indicate a pytorch_eager (onnx2torch) reference bug rather than
        a compiler bug. Verify by comparing against the ONNX spec manually.
Oracle: ORT_ENABLE_ALL output vs pytorch_eager (onnx2torch).
ONNX model: /home/binduan/myspace/trion/trion_results/bug_000355.onnx
Run: python v2_0048_onnxruntime.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os
import numpy as np

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'trion_results', 'bug_000355.onnx')
INPUT = np.array([2.1332738399505615, -1.7384507656097412, 1.5741965770721436, 1.6028645038604736, 1.6152437925338745, -4.275906085968018, -0.48770079016685486, 2.9670255184173584, 2.6751389503479004, -0.3463514447212219, -1.2200853824615479, 2.7324981689453125, -0.7708562016487122, -0.8612374663352966, 1.8587701320648193, -0.8334901332855225, 1.4086374044418335, -1.2274028062820435, 1.1092866659164429, 1.5400869846343994, 0.45020127296447754, -0.609272837638855, -4.943978786468506, 0.7405052781105042, -1.8800934553146362, 4.616432189941406, -1.731791377067566, 1.1187459230422974, -0.8682388067245483, -2.7633464336395264, -0.3073979616165161, -3.2832624912261963, -4.064202785491943, 1.5483165979385376, 1.8810306787490845, -1.2970383167266846, -4.158430099487305, 3.6781110763549805, -2.8917295932769775, -0.32833874225616455, 0.9241238832473755, 1.220413327217102, 0.08464792370796204, -0.8838692903518677, -2.1699106693267822, 1.152305006980896, -0.36805814504623413, -0.10887587070465088, 2.0191709995269775, 0.9024533033370972, -2.4374780654907227, -2.7970829010009766, 1.9022361040115356, 1.0115748643875122, -1.8121594190597534, 0.033265966922044754, -1.0490423440933228, -0.08526397496461868, -3.88356876373291, -0.5891371369361877, 1.1942335367202759, -0.24914273619651794, 1.2295746803283691, 0.0799517035484314], dtype=np.float32).reshape([1, 64])
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
    print(f"Expected (pytorch_eager) at idx 40: {ref.ravel()[40]:.6f}")
    print(f"Got (OnnxRuntime) at idx 40: {result.ravel()[40]:.6f}")  # expected=-0.426204, got=-0.439605
    print(f"rel L2(ORT vs pytorch_eager) = {diff:.6e}")
    if diff > TOLERANCE:
        print("BUG REPRODUCED")
        sys.exit(0)
    print("not reproduced")
    sys.exit(1)

if __name__ == "__main__":
    main()
