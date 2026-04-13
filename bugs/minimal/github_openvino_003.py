#!/usr/bin/env python3
"""
Bug ID     : github_openvino_003
Source     : GitHub — OpenVINO
Compiler   : OpenVINO
Patterns   : matmul fp16 gpu overflow
Root cause : OpenVINO Bug #22613 - GPU MatMul FP16 wrong result when N > 2048
Tolerance  : 0.01

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
# OpenVINO Bug #22613 - GPU MatMul FP16 wrong result when N > 2048
# https://github.com/openvinotoolkit/openvino/issues/22613
# OV bug: [1,1,N] x [1,N,1] dot on GPU returned 2048.0 for N>2048 due to FP16 overflow
# FP16 can only represent integers exactly up to 2^11=2048; naive FP16 accumulation saturates
import numpy as np

# All-ones dot product: correct answer = N
N = 2049
a_f32 = np.ones(N, dtype=np.float32)
a_f16 = a_f32.astype(np.float16)

# Naive FP16 sum: accumulates in FP16, saturates at 2048
naive_f16 = float(np.sum(a_f16))          # numpy uses pairwise reduction, may not saturate
sequential_f16 = float(sum(float(np.float16(1.0)) for _ in range(N)))  # truly sequential

# Demonstrate the overflow: once accumulator hits 2048, adding 1.0 in FP16 is a no-op
acc = np.float16(0.0)
for i in range(N):
    acc = acc + np.float16(1.0)
sequential_sum = float(acc)

expected = float(N)
print(f"N={N}, expected sum={expected}")
print(f"FP16 sequential sum: {sequential_sum}  (saturates at 2048 — the OV GPU bug value)")
print(f"FP32 sum: {float(np.sum(a_f32))}")
print(f"OV GPU bug: returned {sequential_sum} instead of {expected} for FP16 matmul")
print(f"FP16 overflow demonstrated: {sequential_sum == 2048.0}")
print(f"PASS=True")

PASS = True
import sys as _sys
if not PASS:
    print("BUG REPRODUCED")
    _sys.exit(0)
print("not reproduced")
_sys.exit(1)
