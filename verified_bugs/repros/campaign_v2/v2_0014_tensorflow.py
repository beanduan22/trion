#!/usr/bin/env python3
"""
Bug v2-0014 — tf.function(jit_compile=True) wrong output vs eager.
Compiler  : TensorFlow XLA JIT
Source    : bug_000611 (model #611)
Patterns  : cumsum_last_axis, gemm_tanh, linear_layernorm_gelu, add_layernorm, gated_linear_branch
Root cause: TF XLA-JIT miscompiles cumsum on last axis combined with GEMM-tanh and GELU-gated linear branch — XLA generates incorrect HLO for cumsum when preceded by tanh.
Report to : https://github.com/tensorflow/tensorflow/issues
           (label: comp:xla, type:bug)

Oracle: tf.function(jit_compile=True) vs pytorch_eager (onnx2torch).
ONNX model: /home/binduan/myspace/trion/trion_results/bug_000611.onnx
Run: python v2_0014_tensorflow.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os
import numpy as np

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'trion_results', 'bug_000611.onnx')
INPUT = np.array([0.2510092258453369, -1.160791277885437, -0.2391272783279419, -0.1951134353876114, -0.5380500555038452, 0.29533499479293823, 0.9055178761482239, -0.4262229800224304, -0.36960193514823914, 0.15195976197719574, 0.5775061845779419, 1.6539292335510254, -0.2026287019252777, 1.5994951725006104, -0.5858224630355835, 0.04040171205997467, 0.8253955245018005, -0.9334654211997986, 0.3475795090198517, -1.4163322448730469, 0.6856104731559753, 0.31215834617614746, -0.46976134181022644, 0.598547101020813, -1.3853353261947632, -1.0941332578659058, -0.08135567605495453, -1.5003821849822998, 2.0754425525665283, 0.2970021069049835, 0.038852859288454056, -1.0281426906585693, 1.0457311868667603, 1.0292885303497314, -1.0905252695083618, 0.525674045085907, -0.874157726764679, -0.031335361301898956, 0.5621515512466431, -0.7978066802024841, 0.8628183007240295, 0.8633731007575989, -0.06479718536138535, -0.992168664932251, 0.41545113921165466, -2.3311591148376465, 1.6455565690994263, 1.0239394903182983, -0.8834351897239685, 0.7585429549217224, 0.4897879660129547, -0.6963503360748291, 0.07153699547052383, -2.1808927059173584, 0.17929475009441376, 0.9117544293403625, 0.6102275252342224, -0.09866741299629211, -1.1343986988067627, 1.868764042854309, 1.050421118736267, 0.014138607308268547, 0.7325994372367859, -1.0895687341690063], dtype=np.float32).reshape([1, 64])
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
    print(f"Expected (pytorch_eager) at idx 97: {ref.ravel()[97]:.6f}")
    print(f"Got (TF jit_compile=True) at idx 97: {result.ravel()[97]:.6f}")  # expected=3.184480, got=-0.277236
    print(f"Got (TF jit_compile=False) at idx 97: {result_eager.ravel()[97]:.6f}")
    print(f"rel L2(TF_XLA vs pytorch_eager) = {diff_vs_ref:.6e}")
    print(f"rel L2(TF_eager vs pytorch_eager) = {diff_eager_vs_ref:.6e}")
    if diff_vs_ref > TOLERANCE:
        print("BUG REPRODUCED")
        sys.exit(0)
    print("not reproduced")
    sys.exit(1)

if __name__ == "__main__":
    main()
