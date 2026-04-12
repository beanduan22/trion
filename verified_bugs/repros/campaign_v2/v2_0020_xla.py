#!/usr/bin/env python3
"""
Bug v2-0020 — XLA (jax.jit) wrong output vs reference.
Compiler  : XLA / JAX GPU JIT
Source    : bug_000576 (model #576)
Patterns  : ada_layer_norm, layernorm_residual_add, resize_nearest_round_prefer_floor, gap_linear, l2_norm_manual_primitives
Root cause: [SUSPECT: all 6 compilers diverge] XLA may not implement ONNX round_prefer_floor nearest resize correctly.
Report to : https://github.com/google/jax/issues
Note      : XLA bugs require GPU-compiled JAX (cuda-jaxlib).
            On CPU-only JAX, jit ≈ eager and diff is near zero.
NOTE: All 6 tested compilers diverge from pytorch_eager for this model.
        This MAY indicate a pytorch_eager (onnx2torch) reference bug rather than
        a compiler bug. Verify by comparing against the ONNX spec manually.
Oracle: XLA uses pytorch_eager (onnx2torch) as reference via ONNX model.
ONNX model: /home/binduan/myspace/trion/trion_results/bug_000576.onnx
Run: python v2_0020_xla.py  →  exit 0 = BUG REPRODUCED
"""
import sys, os, subprocess

# Delegate to the primary JAX repro file which has the XLA oracle
_ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'unique_2639.py')

if not os.path.exists(_ref):
    print(f"Reference unique_2639.py not found")
    sys.exit(1)

r = subprocess.run([sys.executable, _ref], capture_output=True, timeout=120)
sys.stdout.buffer.write(r.stdout)
sys.stderr.buffer.write(r.stderr)
if r.returncode == 0 and b'BUG REPRODUCED' in r.stdout:
    sys.exit(0)
sys.exit(r.returncode)
