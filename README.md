# XComp — Cross-Compiler Differential Testing for Deep-Learning Compilers

XComp tests ONNX models across multiple deep-learning compiler backends and
reports numerical divergences and crashes.  Each script builds a minimal ONNX
model inline, runs it on every available backend, and prints a comparison table.

## How to run

```bash
pip install onnx onnxruntime openvino onnx2torch torch
```

Run a single reproducer:

```bash
python bugs/raw_bugs/openvino/cross_openvino_conv_add_relu.py
```

```
Compiler           max_diff  Status
------------------------------------------
ORT_opt             0.00004  ok
OpenVINO            0.14621  BUG ***
onnx2torch          0.00004  ok
torch.compile       0.00004  ok
TorchScript         0.00004  ok

BUG REPRODUCED: OpenVINO diverges from ORT_ref (tol=0.01).
```

Run all scripts for a compiler:

```bash
for f in bugs/raw_bugs/openvino/*.py;      do python "$f"; done
for f in bugs/raw_bugs/torch_compile/*.py; do python "$f"; done
for f in bugs/raw_bugs/onnxruntime/*.py;   do python "$f"; done
for f in bugs/raw_bugs/tflite/*.py;        do python "$f"; done
for f in bugs/raw_bugs/xla/*.py;           do python "$f"; done
```

TVM requires its own environment:

```bash
for f in bugs/raw_bugs/tvm/*.py; do
    /path/to/tvm-env/bin/python "$f"
done
```

Exit codes: `0` = bug reproduced, `1` = not reproduced, `2` = missing deps.

## Bug patterns

Each script targets one of 26 distinct root causes.  Full descriptions and
minimal reproducer code are in [`bugs/raw_bugs/ROOT_CAUSES.md`](bugs/raw_bugs/ROOT_CAUSES.md).

| ID    | Pattern                                                          |
|-------|------------------------------------------------------------------|
| RC-01 | Reshape with runtime-tensor shape → FakeTensor crash in torch.compile |
| RC-02 | Conv+BN fusion rounding error (OV/ORT diverge after fusing)     |
| RC-03 | Winograd/tiled GEMM fp32 accumulation order (OpenVINO)          |
| RC-04 | C undefined behaviour: `x >> 64` on x86 returns `x`, not `0`   |
| RC-05 | Slice with full-range indices not folded to identity            |
| RC-06 | Transpose–Transpose cancellation missed by optimizer            |
| RC-07 | Space-to-depth block size mismatch after fusion                 |
| RC-08 | Einsum transpose reorder under tiled execution                  |
| RC-09 | Conv+PReLU channel-wise weight broadcast error                  |
| RC-10 | GroupNorm / LayerNorm precision loss after fusion               |
| RC-11 | Attention logit soft-cap fp32 overflow                          |
| RC-12 | Flatten→Gemm reshape dimension error                            |
| RC-13 | Dilated (ASPP) branch accumulation order                        |
| RC-14 | Pad→Conv boundary element error                                 |
| RC-15 | ReduceL2 last-axis numerical precision                          |
| RC-16 | Reciprocal+Mul constant-folding precision loss                  |
| RC-17 | uint8/int8 Sub/Mul/Add saturates instead of ONNX modular wrap   |
| RC-18 | Gather out-of-bounds: silent 0 instead of error                 |
| RC-19 | Relu(NaN) → 0.0 instead of NaN (IEEE 754 violation)             |
| RC-20 | Exp(NaN) → inf instead of NaN (IEEE 754 violation)              |
| RC-21 | ReduceLogSumExp overflow when inputs are large                  |
| RC-22 | BitShift RIGHT by 64 via C UB (x86 SHRQ masks to 6 bits)        |
| RC-23 | TVM Resize `half_pixel` coordinate shift missing                |
| RC-24 | TVM RoiAlign ignores `coordinate_transformation_mode`           |
| RC-25 | TVM FoldConstant: `inf·X − inf·X` folded to `0` instead of NaN |
| RC-26 | TVM Gelu `approximate="tanh"` silently uses exact erf           |
