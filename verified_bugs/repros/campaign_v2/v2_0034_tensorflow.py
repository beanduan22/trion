#!/usr/bin/env python3
"""
Bug v2-0034 — tf.function(jit_compile=True) wrong output vs eager.
Compiler  : TensorFlow XLA JIT
Source    : bug_001201 (model #1201)
Patterns  : layernorm_temperature, triple_add_residual, self_sub_zero, topk_last_axis_k1, instance_norm_1d
Root cause: [SUSPECT: all 6 compilers diverge] TF XLA-JIT may incorrectly eliminate self-sub-zero (x-x→0) before computing instance norm variance.
Report to : https://github.com/tensorflow/tensorflow/issues
           (label: comp:xla, type:bug)
NOTE: All 6 tested compilers diverge from pytorch_eager for this model.
        This MAY indicate a pytorch_eager (onnx2torch) reference bug rather than
        a compiler bug. Verify by comparing against the ONNX spec manually.
Oracle: tf.function(jit_compile=True) vs pytorch_eager (onnx2torch).
ONNX model: /home/binduan/myspace/trion/trion_results/bug_001201.onnx
Run: python v2_0034_tensorflow.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os
import numpy as np

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'trion_results', 'bug_001201.onnx')
INPUT = np.array([-0.8119166493415833, -0.3905617594718933, -2.129685163497925, -0.5469679236412048, 0.2178342640399933, 0.32410556077957153, 0.0005246857181191444, -0.42530936002731323, -0.695100724697113, 1.9501045942306519, 0.49856626987457275, 0.4276963174343109, 2.106459379196167, -0.19549931585788727, -0.18303067982196808, 0.37609395384788513, 0.44334736466407776, 1.0641059875488281, -0.6673465967178345, 1.6585640907287598, -1.1886591911315918, 0.50031578540802, 0.9379238486289978, -0.6020685434341431, 0.6345017552375793, -1.9413678646087646, -0.11102597415447235, -0.5819206833839417, -0.3662472665309906, 1.1665705442428589, -2.974843978881836, -1.4452849626541138, 1.1058355569839478, -0.8623213768005371, -0.27239328622817993, 1.284787893295288, 0.08016391843557358, 0.023304332047700882, -0.7312196493148804, -0.6601323485374451, -1.7011079788208008, 0.6239184737205505, 1.4479831457138062, -1.1455920934677124, -0.21547667682170868, -0.007334494031965733, -0.20703886449337006, 0.6892555952072144, -0.22440704703330994, 0.046179112046957016, 0.7890960574150085, -0.2384287267923355, 0.22068721055984497, 1.4517382383346558, 0.1458168774843216, 1.1827435493469238, -0.6447463631629944, 0.5556695461273193, -2.9293203353881836, 0.688454270362854, 1.3235301971435547, 0.5274850726127625, -1.3799643516540527, -1.8593478202819824, 0.919698178768158, -0.2219499945640564, 1.594722032546997, 0.5812407732009888, 0.1360965222120285, 0.14420872926712036, 1.5469211339950562, -0.5620183944702148, 0.7350318431854248, 0.6503922343254089, -0.7268590927124023, -1.2683930397033691, -1.5261658430099487, 0.46058815717697144, -0.225577250123024, 1.4939582347869873, 0.2832180857658386, -0.2712605893611908, 0.5950934290885925, -0.8252972960472107, -0.3938058316707611, -0.8702527284622192, -1.1696712970733643, 1.4439834356307983, -0.17125967144966125, 0.19679903984069824, -1.050076961517334, -0.43691322207450867, -1.0084437131881714, -0.3031580150127411, -0.4990861117839813, -0.05296509340405464, -0.9380566477775574, -0.4711599349975586, 0.023113524541258812, -0.20761257410049438, -0.17159166932106018, 1.4564744234085083, 0.11398427188396454, 1.1031811237335205, 0.44683781266212463, -1.296301007270813, 1.221940517425537, -0.14743107557296753, 0.7495607137680054, -0.11658915132284164, -0.13616473972797394, -0.16587889194488525, -0.2821574807167053, -1.6651887893676758, -0.23012931644916534, -0.2846760153770447, 0.06747139245271683, -1.6202729940414429, -1.999261498451233, -0.72735196352005, 0.20242264866828918, 1.1639759540557861, 0.13174930214881897, 0.9048617482185364, 0.26151394844055176, 0.562058687210083, 0.45225203037261963, -0.8543881773948669], dtype=np.float32).reshape([1, 128])
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
        import tensorflow as tf
        import onnx, tf2onnx
        onnx_model = onnx.load(ONNX_PATH)
        tf_model, _ = tf2onnx.convert.from_onnx(onnx_model,
                                                   input_names=["model_input"],
                                                   output_names=None,
                                                   opset=None)

        # Fallback: run via onnx2torch → torch → numpy, then wrap in tf
        # Direct TF XLA path via onnxruntime-training or tf-onnx backend
        import tf2onnx.tfonnx
        inp_tf = tf.constant(INPUT)

        @tf.function(jit_compile=True)
        def run_xla(x):
            # Use saved_model or tf_model here
            raise NotImplementedError("tf2onnx backend needed")

        # Alternative: use onnx-tf if available
        try:
            from onnx_tf.backend import prepare
            tf_rep = prepare(onnx.load(ONNX_PATH))

            @tf.function(jit_compile=True)
            def run_xla(x):
                return tf_rep.graph(x)

            result_xla = run_xla(inp_tf).numpy().ravel()

            @tf.function(jit_compile=False)
            def run_eager(x):
                return tf_rep.graph(x)

            result_eager = run_eager(inp_tf).numpy().ravel()
        except ImportError:
            print("onnx-tf not installed — cannot verify TF XLA path directly")
            sys.exit(2)

    except Exception as e:
        print(f"[error] tensorflow: {e}")
        sys.exit(2)

    diff_vs_ref = _rel_l2(result_xla, ref)
    diff_eager_vs_ref = _rel_l2(result_eager, ref)
    print(f"Expected (pytorch_eager) at idx 0: {ref.ravel()[0]:.6f}")
    print(f"Got (TF jit_compile=True) at idx 0: {result.ravel()[0]:.6f}")  # expected=-0.000045, got=0.000000
    print(f"Got (TF jit_compile=False) at idx 0: {result_eager.ravel()[0]:.6f}")
    print(f"rel L2(TF_XLA vs pytorch_eager) = {diff_vs_ref:.6e}")
    print(f"rel L2(TF_eager vs pytorch_eager) = {diff_eager_vs_ref:.6e}")
    if diff_vs_ref > TOLERANCE:
        print("BUG REPRODUCED")
        sys.exit(0)
    print("not reproduced")
    sys.exit(1)

if __name__ == "__main__":
    main()
