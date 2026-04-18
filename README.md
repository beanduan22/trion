# XComp — Cross-Compiler Bug Collection for Deep-Learning Compilers

A collection of **102 confirmed, minimal reproducers** for bugs found by
differential testing across deep-learning compiler backends.  Each script is
self-contained: it builds the ONNX model inline, runs it on every available
backend, and prints a comparison table showing which compiler diverges.

## Quick start

```bash
pip install onnx onnxruntime openvino onnx2torch torch
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

Exit codes: `0` = bug reproduced, `1` = not reproduced, `2` = missing deps.

## Bug summary

| Compiler         | Bugs | Category                                                        |
|------------------|-----:|-----------------------------------------------------------------|
| OpenVINO CPU     |   57 | GEMM/Conv fp32 precision, NaN propagation, uint8/int8 saturation|
| torch.compile    |   26 | FakeTensor Reshape crash, Inductor bitshift UB, codegen errors  |
| OnnxRuntime      |    7 | Spec violations, bitshift UB (C shift-by-64), SIGFPE            |
| TFLite           |    4 | Attention logit softcap, sub-self-mul-zero, transformer encoder  |
| XLA (TF)         |    4 | MatMul shape, dead-code elimination, Resize coord mode          |
| TVM (0.11.1)     |    4 | Resize half_pixel, RoiAlign coord mode, FoldConstant NaN, Gelu  |
| **Total**        |**102**|                                                                |

## Repository layout

```
bugs/raw_bugs/
├── openvino/      57 scripts
├── torch_compile/ 26 scripts
├── onnxruntime/    7 scripts
├── tflite/         4 scripts
├── xla/            4 scripts
├── tvm/            4 scripts
├── README.md       full bug table with recorded output
└── ROOT_CAUSES.md  26 root causes with minimal reproducer code
```

## Root causes (26 distinct)

See [`bugs/raw_bugs/ROOT_CAUSES.md`](bugs/raw_bugs/ROOT_CAUSES.md) for the full
taxonomy.  Key root causes:

| ID    | Description                                                     | Bugs |
|-------|-----------------------------------------------------------------|-----:|
| RC-01 | onnx2torch Reshape FakeTensor crash under torch.compile         |   22 |
| RC-22 | OpenVINO tiled GEMM/Winograd fp32 accumulation order            |   10 |
| RC-04 | C undefined behaviour: `x >> 64` on x86 returns `x` not `0`    |    3 |
| RC-17 | OpenVINO uint8 Sub/Mul/Add saturates instead of ONNX wrap       |    3 |
| RC-19 | OpenVINO Relu(NaN) → 0.0, should propagate NaN (IEEE 754)       |    1 |
| RC-20 | OpenVINO Exp(NaN) → inf, should propagate NaN                   |    1 |
| RC-25 | TVM FoldConstant folds `inf·X − inf·X` to 0 instead of NaN     |    1 |
| RC-26 | TVM Gelu `approximate="tanh"` silently ignored                  |    1 |
| …     | 18 more — see ROOT_CAUSES.md                                    |      |

## Running all bugs

```bash
# Run all OpenVINO bugs
for f in bugs/raw_bugs/openvino/*.py; do python "$f"; done

# Run a specific compiler group
for f in bugs/raw_bugs/torch_compile/*.py; do python "$f"; done

# TVM bugs require the TVM conda environment
for f in bugs/raw_bugs/tvm/*.py; do
  /path/to/tvm-env/bin/python "$f"
done
```

## Platform

All bugs verified on:
- Python 3.10, Ubuntu 22.04, x86-64
- OnnxRuntime 1.21.0, OpenVINO 2026.0, PyTorch 2.6.0, TVM 0.11.1

## License

MIT
