# Verified Bug Reproducers

37 confirmed compiler bugs found by the Trion fuzzing campaign (500 models tested).

Each file is **fully self-contained** — no project dependencies required.

## Requirements

```bash
pip install numpy onnx onnxruntime torch onnx2torch jax
```

## Run

```bash
python unique_NNNN.py
# Exit 0 = bug reproduced
# Exit 1 = not reproduced
```

## Bug breakdown

| Backend | Count | Description |
|---|---|---|
| JAX/XLA (jax.jit) | 26 | jax.jit diverges from eager on specific op patterns |
| OnnxRuntime (ORT) | 10 | ORT_ENABLE_ALL diverges from pytorch_eager |
| torch.compile | 1 | inductor CUDA divergence from eager |

## Files

| File | Backends | Pattern |
|---|---|---|
| unique_0000.py | xla+tvm | — |
| unique_0001.py | xla+onnxruntime+tvm | — |
| unique_0002.py | xla+tvm | — |
| unique_0003.py | xla+tvm | — |
| unique_0007.py | onnxruntime | — |
| unique_0008.py | torch_compile+onnxruntime | — |
| unique_0010.py | xla+tvm | — |
| unique_0012.py | xla+tvm | — |
| unique_0013.py | onnxruntime | — |
| unique_0016.py | xla+torch_compile+tvm | — |
| unique_0017.py | xla+tvm | — |
| unique_0018.py | onnxruntime | — |
| unique_0022.py | xla+tvm | — |
| unique_0023.py | xla+tvm | — |
| unique_0026.py | xla+onnxruntime+tvm | — |
| unique_0027.py | onnxruntime | — |
| unique_0028.py | xla+onnxruntime+tvm | — |
| unique_0030.py | xla+tvm | — |
| unique_0031.py | onnxruntime | — |
| unique_0032.py | onnxruntime | — |
| unique_0033.py | xla+onnxruntime+tvm | — |
| unique_0034.py | xla+onnxruntime+tvm | — |
| unique_0037.py | xla+tvm | — |
| unique_0038.py | xla+tvm | — |
| unique_0041.py | xla+onnxruntime+tvm | — |
| unique_0042.py | xla+tvm | — |
| unique_0044.py | onnxruntime | — |
| unique_0046.py | xla+tvm | — |
| unique_0047.py | xla+tvm | — |
| unique_0049.py | xla+tvm | — |
| unique_0055.py | xla+tvm | — |
| unique_0056.py | onnxruntime | — |
| unique_0058.py | xla+onnxruntime+tvm | — |
| unique_0064.py | xla+tvm | — |
| unique_0065.py | torch_compile | — |
| unique_0066.py | onnxruntime | — |
| unique_0069.py | xla+onnxruntime+tvm | — |
