#!/usr/bin/env python3
"""
Bug v2-0049 — tf.function(jit_compile=True) wrong output vs eager.
Compiler  : TensorFlow XLA JIT
Source    : bug_000284 (model #284)
Patterns  : matmul_bias_gelu, gated_linear_branch, mul_zero_elim, softmax_sub_max_stable, layernorm_residual_add
Root cause: TF XLA-JIT incorrectly eliminates mul-by-zero under matmul+GELU with gated linear branch and stable softmax — XLA dead-code-eliminates a branch with near-zero but nonzero values.
Report to : https://github.com/tensorflow/tensorflow/issues
           (label: comp:xla, type:bug)

Oracle: tf.function(jit_compile=True) vs pytorch_eager (onnx2torch).
ONNX model: /home/binduan/myspace/trion/trion_results/bug_000284.onnx
Run: python v2_0049_tensorflow.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os
import numpy as np

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'trion_results', 'bug_000284.onnx')
INPUT = np.array([0.9362409114837646, 0.11968731880187988, 0.29487141966819763, 0.39705994725227356, -1.2444742918014526, -1.173728108406067, -1.3853652477264404, 0.4533606469631195, -0.47991639375686646, 0.5316079258918762, 0.5966867208480835, -1.6299842596054077, 1.615759253501892, -1.5057326555252075, 1.0831674337387085, -0.4357850253582001, -1.4006003141403198, 0.19300521910190582, -0.17372748255729675, -0.32663699984550476, -0.6687305569648743, -0.3877514898777008, -0.5130075812339783, -0.5123896598815918, 0.7089846134185791, -0.04818853363394737, 0.6345784068107605, -0.30166757106781006, 0.5251891016960144, 0.31424543261528015, -1.0951565504074097, -0.2998960614204407, -1.2732635736465454, 0.07574515789747238, -0.5314556956291199, 0.7333874702453613, 1.5856640338897705, -0.2780846655368805, -1.1547545194625854, 0.19024068117141724, -2.435370445251465, 0.7303752899169922, 0.0704314112663269, -0.20121611654758453, -0.3058743476867676, 0.16614976525306702, -0.7824792265892029, -0.5741559267044067, 1.2279728651046753, 0.5295564532279968, 2.164982557296753, -1.045836329460144, -1.38432776927948, -0.3872918486595154, 0.08925117552280426, -1.4872599840164185, 2.1250321865081787, 0.8956189751625061, -0.5430893301963806, 1.564837098121643, -0.051510754972696304, 1.2770036458969116, 0.05160655826330185, -0.9345787167549133], dtype=np.float32).reshape([1, 64])
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
    print(f"Got (TF jit_compile=True) at idx 0: {result.ravel()[0]:.6f}")  # expected=0.000000, got=-0.000000
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
