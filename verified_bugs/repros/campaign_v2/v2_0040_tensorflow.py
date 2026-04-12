#!/usr/bin/env python3
"""
Bug v2-0040 — tf.function(jit_compile=True) wrong output vs eager.
Compiler  : TensorFlow XLA JIT
Source    : bug_001069 (model #1069)
Patterns  : gather_reshape, reduce_l2_last, manual_layernorm, matmul_bias_gelu, layernorm_dropout_identity
Root cause: [SUSPECT: all 6 compilers diverge] TF XLA-JIT may miscompile gather-reshape + matmul+bias+GELU + manual layernorm — GELU approximation or gather ordering differs from pytorch_eager.
Report to : https://github.com/tensorflow/tensorflow/issues
           (label: comp:xla, type:bug)
NOTE: All 6 tested compilers diverge from pytorch_eager for this model.
        This MAY indicate a pytorch_eager (onnx2torch) reference bug rather than
        a compiler bug. Verify by comparing against the ONNX spec manually.
Oracle: tf.function(jit_compile=True) vs pytorch_eager (onnx2torch).
ONNX model: /home/binduan/myspace/trion/trion_results/bug_001069.onnx
Run: python v2_0040_tensorflow.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os
import numpy as np

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'trion_results', 'bug_001069.onnx')
INPUT = np.array([-1.6329939365386963, 0.8037511706352234, 0.023493828251957893, 1.3156461715698242, 0.8124868869781494, -1.1372724771499634, 0.7371470928192139, 2.8884310722351074, 1.3455729484558105, -0.433082640171051, -1.945820689201355, -0.10299476981163025, -0.38271287083625793, 1.5492949485778809, 1.932789921760559, -0.39534372091293335, 1.4016050100326538, -0.028126787394285202, 0.3759615421295166, -1.4285181760787964, 0.39250603318214417, 0.5443388223648071, -0.39636048674583435, 1.0185961723327637, 0.12618191540241241, -0.11381352692842484, 0.2940337061882019, -0.6314786076545715, -0.3105968236923218, 1.01949143409729, -1.291857123374939, -1.2713853120803833, -0.38416624069213867, 0.3345392942428589, -0.971969485282898, -1.0642354488372803, -0.8123127818107605, 0.46304842829704285, 1.5911073684692383, 0.5359366536140442, -1.1230653524398804, -1.088464617729187, -0.6919722557067871, -1.3264386653900146, -0.7881543636322021, 1.1937488317489624, 0.10810521245002747, 0.5777326822280884, 0.5963307619094849, 1.1003309488296509, 1.5420117378234863, -2.5095083713531494, 1.215969204902649, 0.2754155993461609, 1.0586698055267334, -0.755833625793457, -0.9228277206420898, -2.4412620067596436, 2.4278435707092285, 0.843078076839447, -0.37668487429618835, -0.27715831995010376, 0.5172286629676819, 0.29543352127075195, -2.3368191719055176, 0.41723203659057617, -0.25788992643356323, 0.24039198458194733, 1.0160149335861206, -1.619115948677063, -0.07945717126131058, -0.18987929821014404, 1.1329642534255981, -1.0588479042053223, -0.3205873668193817, -0.29058918356895447, 0.7046172022819519, -0.09016967564821243, 0.8931432962417603, 1.851197600364685, -0.7365316152572632, -0.5055485963821411, 0.469197541475296, -2.3997035026550293, -0.7600677013397217, -0.4162605404853821, 1.968517780303955, 0.3443475365638733, -0.4175511598587036, -0.7619228363037109, -2.668875217437744, 1.0292154550552368, -0.9523301124572754, -1.1511110067367554, 0.16491350531578064, -1.5983095169067383, 0.39732709527015686, -0.9989608526229858, -1.0005768537521362, 1.9441224336624146, -0.6367359161376953, -0.030720485374331474, -0.4416583478450775, -0.433059960603714, -0.6072307825088501, -0.42543309926986694, -1.0711758136749268, -1.0549921989440918, -0.015558246523141861, -0.6742048263549805, -0.4053923785686493, 0.46385055780410767, 0.37388095259666443, -0.014759502373635769, 0.8982767462730408, -0.7153335213661194, 1.0816267728805542, 1.8620680570602417, 1.0925326347351074, 0.08724025636911392, 0.6945616006851196, 0.581707775592804, -0.024204444140195847, 0.6035202145576477, 1.1240304708480835, 0.907745897769928, -0.8940295577049255, 0.5865160226821899], dtype=np.float32).reshape([1, 128])
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
    print(f"Expected (pytorch_eager) at idx 48: {ref.ravel()[48]:.6f}")
    print(f"Got (TF jit_compile=True) at idx 48: {result.ravel()[48]:.6f}")  # expected=-0.001027, got=0.000000
    print(f"Got (TF jit_compile=False) at idx 48: {result_eager.ravel()[48]:.6f}")
    print(f"rel L2(TF_XLA vs pytorch_eager) = {diff_vs_ref:.6e}")
    print(f"rel L2(TF_eager vs pytorch_eager) = {diff_eager_vs_ref:.6e}")
    if diff_vs_ref > TOLERANCE:
        print("BUG REPRODUCED")
        sys.exit(0)
    print("not reproduced")
    sys.exit(1)

if __name__ == "__main__":
    main()
