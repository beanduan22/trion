# Verified Bug Reproducers

99 confirmed compiler bugs found by the Trion fuzzing campaign.

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
| JAX/XLA + TVM (jax.jit) | 88 | jax.jit/XLA diverges from pytorch_eager reference |
| OnnxRuntime (ORT) | 1 | ORT_ENABLE_ALL diverges from ORT_DISABLE_ALL |
| torch.compile | 10 | torch.compile(inductor) diverges from eager |

**Comparison strategies used:**
- JAX bugs: native JAX model vs embedded pytorch_eager reference output (88 bugs), or jax.jit vs jax.disable_jit (4 of the 88)
- ORT bugs: ORT with full optimization vs ORT with no optimization
- torch.compile bugs: compiled inductor vs eager pytorch

## Files

| File | Backends | Ops |
|---|---|---|
| unique_0001.py | xla+tvm | Add, BatchNorm, Conv, ... |
| unique_0012.py | xla+tvm | — |
| unique_0016.py | xla+torch_compile+tvm | — |
| unique_0042.py | xla+tvm | — |
| unique_0530.py | torch_compile | — |
| unique_0553.py | torch_compile | — |
| unique_0570.py | torch_compile | — |
| unique_0632.py | onnxruntime | — |
| unique_1093.py | torch_compile | — |
| unique_1117.py | torch_compile | — |
| unique_1184.py | torch_compile | — |
| unique_1259.py | xla+tvm | — |
| unique_1294.py | xla+tvm | — |
| unique_1306.py | xla+tvm | — |
| unique_1314.py | xla+tvm | — |
| unique_1357.py | xla+tvm | — |
| unique_1401.py | torch_compile | — |
| unique_1413.py | xla+tvm | — |
| unique_1433.py | xla+tvm | — |
| unique_1457.py | xla+tvm | — |
| unique_1524.py | xla+tvm | — |
| unique_1599.py | xla+tvm | — |
| unique_1634.py | xla+tvm | — |
| unique_1646.py | xla+tvm | — |
| unique_1654.py | xla+tvm | — |
| unique_1697.py | xla+tvm | — |
| unique_1741.py | xla+tvm | — |
| unique_1753.py | xla+tvm | — |
| unique_1773.py | xla+tvm | — |
| unique_1790.py | xla+tvm | — |
| unique_1811.py | xla+tvm | — |
| unique_1830.py | xla+tvm | — |
| unique_1891.py | xla+tvm | — |
| unique_1894.py | xla+tvm | — |
| unique_1913.py | xla+tvm | — |
| unique_1917.py | xla+tvm | — |
| unique_1934.py | xla+tvm | — |
| unique_1939.py | xla+tvm | — |
| unique_1976.py | xla+tvm | — |
| unique_1977.py | torch_compile | — |
| unique_1996.py | xla+tvm | — |
| unique_2010.py | xla+tvm | — |
| unique_2014.py | xla+tvm | — |
| unique_2017.py | xla+tvm | — |
| unique_2020.py | xla+tvm | — |
| unique_2070.py | xla+tvm | — |
| unique_2087.py | xla+tvm | — |
| unique_2162.py | xla+tvm | — |
| unique_2183.py | xla+tvm | — |
| unique_2197.py | xla+tvm | — |
| unique_2209.py | xla+tvm | — |
| unique_2213.py | xla+tvm | — |
| unique_2217.py | xla+tvm | — |
| unique_2231.py | xla+tvm | — |
| unique_2255.py | xla+tvm | — |
| unique_2260.py | xla+tvm | — |
| unique_2304.py | xla+tvm | — |
| unique_2316.py | xla+tvm | — |
| unique_2329.py | xla+tvm | — |
| unique_2336.py | xla+tvm | — |
| unique_2353.py | xla+tvm | — |
| unique_2374.py | xla+tvm | — |
| unique_2393.py | xla+tvm | — |
| unique_2411.py | xla+tvm | — |
| unique_2417.py | xla+tvm | — |
| unique_2454.py | xla+tvm | — |
| unique_2457.py | xla+tvm | — |
| unique_2476.py | xla+tvm | — |
| unique_2480.py | xla+tvm | — |
| unique_2497.py | xla+tvm | — |
| unique_2500.py | xla+tvm | — |
| unique_2502.py | xla+tvm | — |
| unique_2525.py | xla+tvm | — |
| unique_2539.py | xla+tvm | — |
| unique_2540.py | xla+tvm | — |
| unique_2559.py | xla+tvm | — |
| unique_2560.py | xla+tvm | — |
| unique_2561.py | xla+onnxruntime+tvm | — |
| unique_2563.py | xla+tvm | — |
| unique_2570.py | xla+tvm | — |
| unique_2572.py | xla+tvm | — |
| unique_2576.py | xla+torch_compile+tvm | — |
| unique_2577.py | xla+tvm | — |
| unique_2582.py | xla+tvm | — |
| unique_2583.py | xla+tvm | — |
| unique_2586.py | xla+onnxruntime+tvm | — |
| unique_2588.py | xla+onnxruntime+tvm | — |
| unique_2590.py | xla+tvm | — |
| unique_2593.py | xla+onnxruntime+tvm | — |
| unique_2594.py | xla+onnxruntime+tvm | — |
| unique_2597.py | xla+tvm | — |
| unique_2598.py | xla+tvm | — |
| unique_2601.py | xla+onnxruntime+tvm | — |
| unique_2602.py | xla+tvm | — |
| unique_2606.py | xla+tvm | — |
| unique_2607.py | xla+tvm | — |
| unique_2609.py | xla+tvm | — |
| unique_2615.py | xla+tvm | — |
| unique_2618.py | xla+onnxruntime+tvm | — |
