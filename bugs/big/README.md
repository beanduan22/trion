# XComp — New Bugs (witness-based)

Three reproducers for the **17 lower-severity findings** of the
1000-model campaign that do **not** reduce to a small hand-built graph.
Each script ships with the exact campaign witness ONNX (and, where the
divergence is input-dependent, a `.input.npy` of the input that
triggered it) so the bug reproduces deterministically.

> Looking for the high-severity findings? See **`bugs/new_found/`** —
> 5 self-contained scripts that build the buggy graph in-memory.

## Coverage

| # | Script | Bugs | Why witness-based |
|---|--------|------|-------------------|
| rc_05 | `rc_05_xla_int32_cast_chain_drift.py`         | 529, 579, 635, 796, 928, 990 | XLA's algebraic simplifier only elides the `Cast(int32)→Cast(fp32)` pair when surrounded by enough matmul context |
| rc_07 | `rc_07_openvino_transformer_layer_segfault.py` | crashes 53, 241, 412, 414    | OV CPU plugin SIGSEGV is shape/structure-sensitive; even a 25-node hand-built transformer block does not crash |
| rc_08 | `rc_08_small_numerical_drift_cluster.py`       | 210, 214, 224, 297, 392, 413, 585 | Drift is the cumulative effect of long matmul/LN chains — minimal graphs never accumulate enough to cross 1% |

In total **13 divergence/precision bugs + 4 OpenVINO crashes = 17 issues**.

## Layout

```
bugs/big/
├── README.md                                       (this file)
├── rc_05_xla_int32_cast_chain_drift.py             (~ 90 lines)
├── rc_07_openvino_transformer_layer_segfault.py    (~ 90 lines)
├── rc_08_small_numerical_drift_cluster.py          (~ 90 lines)
└── witnesses/
    ├── bug_000{210,214,224,297,392,413,529,579,
    │            585,635,796,928,990}.onnx          (13 ONNX models)
    ├── bug_000XXX.input.npy                        (13 recorded inputs)
    └── compiler_crashes/
        └── crash_000{053,241,412,414}.onnx         (4 OV-crash models)
```

Total payload ≈ 7.7 MB.

## How to run

From the **XComp repo root** (so `trion/` is importable):

```bash
# rc_05 — pick any witness (default = bug 529)
PYTHONPATH=. python bugs/big/rc_05_xla_int32_cast_chain_drift.py
PYTHONPATH=. python bugs/big/rc_05_xla_int32_cast_chain_drift.py 635
PYTHONPATH=. python bugs/big/rc_05_xla_int32_cast_chain_drift.py 990

# rc_07 — pick any of the four OV crashes (default = 241)
PYTHONPATH=. python bugs/big/rc_07_openvino_transformer_layer_segfault.py
PYTHONPATH=. python bugs/big/rc_07_openvino_transformer_layer_segfault.py 053

# rc_08 — sweeps all 7 small-drift witnesses
PYTHONPATH=. python bugs/big/rc_08_small_numerical_drift_cluster.py
```

Each script exits with **0 = bug reproduced**, 1 = ran cleanly, 2 = backend
or witness missing.

## Per-script reference output

### rc_05 — XLA int32 cast-roundtrip drift

```
Witness: bug_529  shape=[1, 16, 256]  nodes=15
Cast nodes (roundtrip pair): [10, 11]
ORT  first 5 : [ 0.00065603  0.0105158  -0.00170846  0.01040263  0.00097912]
XLA  first 5 : [ 0.00065605  0.01051595 -0.00170804  0.01040277  0.00097885]
rel_L2(XLA, ORT) = 0.028784   max_abs_diff = 9.9962e-03
tolerance        = 0.01
worst positions:
  [ 417] ORT= +0.000103  XLA= -0.000093  diff=+1.956e-04
  [ 326] ORT= +0.029668  XLA= +0.019687  diff=+9.982e-03
  [1490] ORT= +0.019462  XLA= +0.029458  diff=-9.996e-03

BUG REPRODUCED — XLA disagrees with ORT past the campaign's 1% tolerance.
```

### rc_07 — OpenVINO SIGSEGV

```
Witness: .../crash_000241.onnx  shape=[1, 32, 256]  nodes=31
ORT (correct): out shape=(1, 32, 128)  max_abs=4.7840e+00  any NaN=False
OV subprocess return code = -11  (SIGSEGV = -11)

BUG REPRODUCED — OpenVINO segfaults compiling the witness graph,
while ORT produced a finite output for the same model and input.
```

### rc_08 — small-drift cluster (sweeps 7 witnesses)

```
 bug        backend   campaign    this run    max_abs verdict
 210            tvm     0.0140    0.013744  2.040e-01 REPRODUCED
 214            xla     0.0450    0.044856  1.000e-02 REPRODUCED
 224  torch_compile     0.0150    0.014543  3.575e-01 REPRODUCED
 297            xla     0.0100    0.010315  5.003e-01 REPRODUCED
 392    torchscript     0.0340    0.034203  7.371e-02 REPRODUCED
 413    torchscript     0.0180    0.017981  1.880e-03 REPRODUCED
 585            tvm 145303.0000 145303.460995 2.638e-04 REPRODUCED

Reproduced: 7/7 (with 0 skipped)
```

## Prerequisites

| Backend | Used by | Install |
|---------|---------|---------|
| onnxruntime | all  | `pip install onnxruntime` |
| openvino    | rc_07| `pip install openvino` |
| jax / jaxlib| rc_05, rc_08 (XLA cases)| `pip install jax jaxlib` |
| torch / onnx2torch | rc_08 (TS, TC cases) | `pip install torch onnx2torch` |
| Apache TVM  | rc_08 (TVM cases) | Python-3.10 worker — see `trion/oracle/tvm_backend.py` |

## Bug → reproducer cross-reference

```
bugs 529, 579, 635, 796, 928, 990            →  rc_05
crashes 053, 241, 412, 414  (OpenVINO)       →  rc_07
bugs 210, 214, 224, 297, 392, 413, 585       →  rc_08
```
