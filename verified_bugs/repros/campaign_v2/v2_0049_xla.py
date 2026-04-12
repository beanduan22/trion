#!/usr/bin/env python3
"""
Bug v2-0049 — XLA (jax.jit) wrong output vs reference.
Compiler  : XLA / JAX GPU JIT
Source    : bug_000284 (model #284)
Patterns  : matmul_bias_gelu, gated_linear_branch, mul_zero_elim, softmax_sub_max_stable, layernorm_residual_add
Root cause: XLA GPU JIT incorrectly eliminates mul-by-zero under matmul+GELU with gated linear branch and stable softmax — XLA dead-code-eliminates a near-zero branch.
Report to : https://github.com/google/jax/issues
Note      : XLA bugs require GPU-compiled JAX (cuda-jaxlib).
            On CPU-only JAX, jit ≈ eager and diff is near zero.

Oracle: XLA uses pytorch_eager (onnx2torch) as reference via ONNX model.
ONNX model: /home/binduan/myspace/trion/trion_results/bug_000284.onnx
Run: python v2_0049_xla.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os, subprocess

# Delegate to the primary JAX repro file which has the XLA oracle
_ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'unique_2668.py')

if not os.path.exists(_ref):
    print(f"Reference unique_2668.py not found")
    sys.exit(1)

r = subprocess.run([sys.executable, _ref], capture_output=True, timeout=120)
sys.stdout.buffer.write(r.stdout)
sys.stderr.buffer.write(r.stderr)
if r.returncode == 0 and b'BUG REPRODUCED' in r.stdout:
    sys.exit(0)
sys.exit(r.returncode)
