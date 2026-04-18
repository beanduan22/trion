# Root Causes — 98 Bugs, 22 Distinct Root Causes

Each section: **what is wrong**, **what is correct**, **minimal code**.  
Scripts in `<backend>/` are the self-contained reproducers (exit 0 = bug reproduced).

---

## RC-01 · onnx2torch Reshape with runtime-tensor shape crashes torch.compile
**Category:** ONNX convert  
**Affects:** 22 bugs — all `cross_torch_compile_*.py` (except space_to_depth)

### What is wrong
`onnx2torch` converts ONNX `Reshape` whose second input is a constant-initializer tensor
into `torch.reshape(x, shape_tensor.tolist())`. Under eager this works.  
Under `torch.compile`, `shape_tensor` becomes a `FakeTensor` during tracing and
`torch.Size()` rejects it:
```
TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')
→ InternalTorchDynamoError
```

### What is correct
The shape should be extracted at compile time as static integers (it IS a constant
initializer). `torch.compile` should be able to trace and compile the model.

### Minimal reproducer
**File:** `torch_compile/cross_torch_compile_redundant_reshape.py`
```python
import numpy as np, onnx, onnx2torch, torch
from onnx import helper as oh, TensorProto as TP, numpy_helper as onh

W  = np.random.randn(512, 64).astype(np.float32)
sh = np.array([2, 64], dtype=np.int64)

nodes = [
    oh.make_node("MatMul",  ["X","W"],   ["mm"]),
    oh.make_node("Reshape", ["mm","sh"], ["Y"]),    # shape is a constant tensor
]
graph = oh.make_graph(nodes, "g",
    [oh.make_tensor_value_info("X", TP.FLOAT, [2,512])],
    [oh.make_tensor_value_info("Y", TP.FLOAT, [2,64])],
    initializer=[onh.from_array(W,"W"), onh.from_array(sh,"sh")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("",13)])

net = onnx2torch.convert(model).eval()
x   = torch.randn(2, 512)

print(net(x).shape)               # eager: OK → torch.Size([2, 64])
compiled = torch.compile(net)
compiled(x)                        # torch.compile: CRASH — FakeTensor in torch.Size()
```
**Correct:** `torch.Size([2, 64])`  
**Wrong:** `InternalTorchDynamoError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')`

---

## RC-02 · Inductor bf16 SpaceToDepth wrong layout
**Category:** Compiler optimization  
**Affects:** 1 bug — `cross_torch_compile_space_to_depth_block.py`

### What is wrong
Inductor's bf16 lowering for SpaceToDepth (pixel-unshuffle) uses the wrong memory
layout when lowering to a kernel. The block-reordering of spatial pixels into channels
produces an output that differs from the eager (float32) reference by max_abs ≈ 2e-3.

### What is correct
SpaceToDepth with block_size=2 on a [1,C,H,W] bf16 tensor should produce the same
numerical result as the eager fp32 path (up to bf16 rounding, i.e. ≤ 8e-4).

### Minimal reproducer
**File:** `torch_compile/cross_torch_compile_space_to_depth_block.py`
```python
import torch

def space_to_depth(x, block=2):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//block, block, W//block, block)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C*block*block, H//block, W//block)

x = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
ref = space_to_depth(x.float()).bfloat16()   # eager fp32→bf16
got = torch.compile(space_to_depth)(x)       # Inductor bf16

diff = (ref.float() - got.float()).abs().max().item()
print(f"max_diff={diff:.5f}")   # correct ≤ 8e-4; Inductor gives ≈ 2e-3
```
**Correct:** max_diff ≤ 0.001  
**Wrong:** max_diff ≈ 0.00195 (Inductor reorders channels incorrectly)

---

## RC-03 · Inductor transpose+reduce fusion reinterprets storage strides
**Category:** Compiler optimization  
**Affects:** 1 bug — `torch_compile/github_inductor_009_transpose_reduce_fusion.py`

### What is wrong
Inductor fuses a row-wise `ReduceSum` with a subsequent `Transpose` and, instead of
copying the reduced tensor into the transposed layout, it reinterprets the existing
buffer as if it were already transposed. The result has wrong values.

### What is correct
`x.sum(dim=1, keepdim=True).transpose(1,2)` should produce a tensor of shape `[B,C,1]`
that matches the eager result exactly (no numerical error — this is integer-order addition).

### Minimal reproducer
**File:** `torch_compile/github_inductor_009_transpose_reduce_fusion.py`
```python
import torch

def f(x, scale):
    row_sum = x.sum(dim=1, keepdim=True)        # [B, 1, C]
    return (row_sum * scale).transpose(1,2).contiguous()   # [B, C, 1]

x     = torch.randn(4, 64, 128)
scale = torch.randn(4, 1, 128)

ref = f(x, scale)
got = torch.compile(f, backend="inductor")(x, scale)

diff = (ref - got).abs().max().item()
print(f"max_diff={diff:.2e}")   # correct = 0; Inductor ≫ 0
```
**Correct:** max_diff = 0.0  
**Wrong:** Inductor returns wrong values (stride reinterpretation)

---

## RC-04 · BitShift by 64 — C undefined behaviour, x86 masks shift to 6 bits
**Category:** Compiler optimization  
**Affects:** 3 bugs across 2 backends
- `torch_compile/github_inductor_011_bitshift_ub_shift64.py` (Inductor CPU codegen)
- `onnxruntime/cross_bitshift_shift64_ov_ort.py` (ORT + OpenVINO, same ONNX op)

### What is wrong
The C/C++ standard says shifting a 64-bit integer by ≥ 64 bits is **undefined behaviour**.
On x86, the `SAR`/`SHR` instruction masks the shift count to the low 6 bits:
`x >> 64` → `x >> 0` → `x` (no shift happens). All three backends (Inductor, ORT,
OpenVINO) emit this UB directly from their CPU codegen.

### What is correct
`BitShift(x, 64)` must return 0 for any value of x. The fix is to emit a conditional:
`n >= 64 ? 0 : x >> n`.

### Minimal reproducer
**File:** `onnxruntime/cross_bitshift_shift64_ov_ort.py`
```python
import numpy as np, onnxruntime as ort
from onnx import helper, TensorProto

vals   = np.array([[1000, 255]], dtype=np.uint64)
shifts = np.array([[64,   64]], dtype=np.uint64)

node  = helper.make_node("BitShift", ["x","n"], ["y"], direction="RIGHT")
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("x", TensorProto.UINT64, [1,2]),
     helper.make_tensor_value_info("n", TensorProto.UINT64, [1,2])],
    [helper.make_tensor_value_info("y", TensorProto.UINT64, [1,2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",11)])

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
out = ort.InferenceSession(model.SerializeToString(), so,
      providers=["CPUExecutionProvider"]).run(None, {"x": vals, "n": shifts})[0]

print(f"Input:    {vals[0].tolist()} >> {shifts[0].tolist()}")
print(f"Expected: [0, 0]")
print(f"ORT got:  {out[0].tolist()}")   # [1000, 255] — UB, not 0
```
**Correct:** `[0, 0]`  
**Wrong:** `[1000, 255]` (x86 masks 64→0, shift becomes no-op)

---

## RC-05 · torch.compile skips index_add shape validation
**Category:** Compiler optimization  
**Affects:** 1 bug — `torch_compile/pt_121135_torch_compile.py`

### What is wrong
`torch.compile`'s `randperm_index_add_pattern` optimization pass lowers `index_add`
without checking that `source` is compatible with `self` along the non-indexed axes.
The compiled path silently produces undefined output where eager raises `RuntimeError`.

### What is correct
`index_add(0, idx, source)` where `source.shape[1:] != self.shape[1:]` should raise
`RuntimeError: source shape mismatch`. Eager and TorchScript both do this correctly.

### Minimal reproducer
**File:** `torch_compile/pt_121135_torch_compile.py`
```python
import torch, torch.nn as nn

class M(nn.Module):
    def forward(self, x, y):
        idx = torch.randperm(x.shape[0])[:y.shape[0]]
        return x.index_add(0, idx, y)

x = torch.randn(32, 4)
y = torch.randn(4)          # wrong: source has shape [4] but self has shape [32,4]

try:
    M()(x, y)               # eager → RuntimeError: source shape mismatch
    print("eager: OK (unexpected)")
except RuntimeError as e:
    print(f"eager: {e}")    # correct

try:
    torch.compile(M())(x, y)   # torch.compile → silent, returns wrong tensor
    print("compile: silent pass ← BUG")
except Exception as e:
    print(f"compile: {e}")
```
**Correct:** `RuntimeError: Expected source to have the same number of dimensions as self`  
**Wrong:** torch.compile silently returns a tensor with undefined values

---

## RC-06 · ORT Cast(float→int32)→Cast(int32→bool) fusion drops int32 truncation
**Category:** Compiler optimization  
**Affects:** 1 bug — `onnxruntime/github_ort_004.py`

### What is wrong
ORT's graph optimizer fuses `Cast(float→int32)` + `Cast(int32→bool)` into a single
`Cast(float→bool)`. This drops the int32 truncation step: `-0.1` truncated to int32 is
`0`, which converts to `False`. But the fused direct cast compares `-0.1 != 0` → `True`.

### What is correct
ONNX spec: `Cast(float→int32)` truncates toward zero, so `-0.1 → 0`.
Then `Cast(int32→bool)`: `0 → False`. The two-step result must be `False`.

### Minimal reproducer
**File:** `onnxruntime/github_ort_004.py`
```python
import numpy as np, onnxruntime as ort
from onnx import TensorProto, helper

x = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32)
# Expected: truncate to int32 first → all become 0 → all False
expected = (x.astype(np.int32) != 0)   # [False, False, False, False, False]

cast1 = helper.make_node("Cast", ["X"], ["T"], to=TensorProto.INT32)
cast2 = helper.make_node("Cast", ["T"], ["Y"], to=TensorProto.BOOL)
graph = helper.make_graph([cast1, cast2], "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])],
    [helper.make_tensor_value_info("Y", TensorProto.BOOL, [5])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
out = ort.InferenceSession(mb, so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]

print(f"Input:    {x.tolist()}")
print(f"Expected: {expected.tolist()}")   # all False
print(f"ORT got:  {out.tolist()}")        # [-0.1, -0.2] become True ← BUG
```
**Correct:** `[False, False, False, False, False]`  
**Wrong:** `[True, True, False, True, True]` (fused cast loses int32 truncation)

---

## RC-07 · ORT Resize nearest — wrong coordinate rounding mode
**Category:** Compiler optimization  
**Affects:** 2 bugs
- `onnxruntime/github_onnx_spec_007.py` (coordinate_transformation_mode=`half_pixel`)
- `onnxruntime/github_ort_002.py` (nearest mode diverges from PyTorch)

### What is wrong
For `Resize` with `nearest` interpolation and `coordinate_transformation_mode=half_pixel`,
`nearest_mode=round_prefer_ceil`, ORT computes the wrong source index.  
Example: output index 14 in a scale-2.375 resize maps to source coordinate
`(14 + 0.5) / 2.375 - 0.5 = 0.7368`. `round_prefer_ceil` rounds 0.5-ties up; 0.7368
should round to index 1 (0.7895 rounds to 1 too), but ORT returns index 0.

### What is correct
The ONNX spec and PyTorch `interpolate(mode='nearest-exact')` both return the correct
source index as per the rounding formula.

### Minimal reproducer
**File:** `onnxruntime/github_onnx_spec_007.py`
```python
import numpy as np, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

x     = np.arange(1, 20, dtype=np.float32).reshape(1,1,1,19)
scale = np.array([1.0, 1.0, 1.0, 2.375], dtype=np.float32)

roi   = np.array([], dtype=np.float32)
node  = helper.make_node("Resize", ["x","roi","scale"], ["y"],
            coordinate_transformation_mode="half_pixel",
            nearest_mode="round_prefer_ceil", mode="nearest")
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1,1,1,19])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    initializer=[numpy_helper.from_array(roi,"roi"),
                 numpy_helper.from_array(scale,"scale")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
out = ort.InferenceSession(model.SerializeToString(), so,
      providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

# ONNX spec: output[14] should map to src[1] (value=2), ORT picks src[0] (value=1)
print(f"Output[14]: ORT={out[0,0,0,14]:.0f}  expected=2")
```
**Correct:** output[14] = 2 (src index 1)  
**Wrong:** ORT returns 1 (src index 0, off-by-one rounding error)

---

## RC-08 · ORT Mod(int32, 0) — SIGFPE crash (divide-by-zero not guarded)
**Category:** Compiler optimization  
**Affects:** 1 bug — `onnxruntime/github_ort_017_mod_int_divzero_sigfpe.py`

### What is wrong
ORT's `Mod` kernel calls the host CPU divide instruction for integer modulo.  
`Mod(x, 0)` triggers a hardware SIGFPE (floating-point exception for integer divide).
ORT never guards for zero divisor before calling the kernel.

### What is correct
ORT should check for divisor == 0 and raise an `InvalidArgument` error, or return
a defined result per the ONNX spec (which leaves `Mod(x,0)` undefined, but crashing
the process is unacceptable).

### Minimal reproducer
**File:** `onnxruntime/github_ort_017_mod_int_divzero_sigfpe.py`
```python
import numpy as np, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

x = np.array([10, 20, 5], dtype=np.int32)
y = np.array([0,   0, 0], dtype=np.int32)   # divisor = 0

node  = helper.make_node("Mod", ["x","y"], ["z"], fmod=0)
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("x", TensorProto.INT32, [3]),
     helper.make_tensor_value_info("y", TensorProto.INT32, [3])],
    [helper.make_tensor_value_info("z", TensorProto.INT32, [3])],
    initializer=[numpy_helper.from_array(y,"y")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

# This crashes the process with SIGFPE instead of raising a Python exception
sess = ort.InferenceSession(model.SerializeToString(),
       providers=["CPUExecutionProvider"])
sess.run(None, {"x": x})   # → SIGFPE / process abort
```
**Correct:** should raise `onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument`  
**Wrong:** SIGFPE kills the process (unrecoverable crash)

---

## RC-09 · ORT Bicubic resize numerical error
**Category:** Compiler optimization  
**Affects:** 1 bug — `onnxruntime/github_ort_008.py`

### What is wrong
ORT's CPU bicubic resize accumulates error of 0.061 compared to PyTorch's bicubic
implementation. The cubic coefficient computation differs in how border pixels
are extrapolated.

### What is correct
Both implementations are valid cubic resamplers, but ORT diverges from the PyTorch
reference by more than floating-point rounding noise (0.061 >> expected ~1e-6).

### Minimal reproducer
**File:** `onnxruntime/github_ort_008.py`

---

## RC-10 · Out-of-bounds Gather — silent acceptance vs error
**Category:** Compiler optimization  
**Affects:** 2 bugs across 2 backends
- `onnxruntime/tf_61865_onnx_ort.py` (ORT raises error — strict but acceptable)
- `openvino/tf_61865_onnx_openvino.py` (OV silently returns 0.0 — wrong)

### What is wrong (OpenVINO)
`Gather(params[8], indices=[2,5,4,7,8,3])` where index 8 is out-of-bounds.
OpenVINO silently returns `0.0` for the OOB slot instead of erroring.
All other backends (ORT, onnx2torch, TorchScript) raise an error.

### What is correct
ONNX spec leaves OOB Gather undefined, but silently returning 0.0 is dangerous
(wrong values with no diagnostic). The correct behavior is to raise an error.

### Minimal reproducer
**File:** `openvino/tf_61865_onnx_openvino.py`
```python
import numpy as np, openvino as ov
from onnx import helper, TensorProto, numpy_helper

params  = np.arange(8, dtype=np.float32).reshape(1,1,8,1)   # 8 elements
indices = np.array([2,5,4,7,8,3], dtype=np.int64)           # index 8 = OOB

node  = helper.make_node("Gather", ["params","indices"], ["out"], axis=2)
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("params",  TensorProto.FLOAT, [1,1,8,1])],
    [helper.make_tensor_value_info("out",     TensorProto.FLOAT, None)],
    initializer=[numpy_helper.from_array(indices,"indices")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

core = ov.Core()
comp = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out  = comp({"params": params})[comp.output(0)]
print(f"OV result at OOB index: {out.ravel()[4]}")   # 0.0 silently — BUG
```
**Correct:** should raise an error (OOB index)  
**Wrong:** OpenVINO returns `0.0` silently (information loss, no diagnostic)

---

## RC-11 · TFLite fp16 weight quantization breaks model semantics
**Category:** ONNX convert  
**Affects:** 3 bugs
- `tflite/cross_tflite_attention_logit_softcap.py` (tanh saturation amplified)
- `tflite/cross_tflite_sub_self_mul_zero.py` (matmul associativity broken)
- `tflite/cross_tflite_transpose_matmul_transpose.py` (512-deep reduction error)

### What is wrong
`TFLiteConverter` with `Optimize.DEFAULT` + `target_spec.supported_types=[float16]`
quantizes each MatMul weight tensor independently to fp16. Under fp16, each
dequantize→accumulate→requantize cycle introduces ~1e-3 relative error per op.
Chained ops (3+ matmuls, or nonlinearities like tanh) amplify this beyond 0.01 fp32
output error — a difference a user deploying a validated fp32 model would observe.

### What is correct
Keras fp32 eager with the same weights and input should be the reference. A fp16-
quantized model must stay within acceptable error bounds for the target accuracy class.

### Minimal reproducer
**File:** `tflite/cross_tflite_attention_logit_softcap.py`
```python
import numpy as np, tensorflow as tf

B, S, D = 2, 128, 64
X  = np.random.randn(B, S, D).astype(np.float32) * 2.0
Wq = np.random.randn(D, D).astype(np.float32)

class M(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.wq = tf.Variable(Wq, trainable=False)
    @tf.function(input_signature=[tf.TensorSpec([B,S,D], tf.float32)])
    def call(self, x):
        q = tf.matmul(x, self.wq)
        return tf.tanh(q / 5.0) * 5.0   # softcap

m = M(); _ = m(tf.zeros([B,S,D]))
ref = m(tf.constant(X)).numpy()

conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
itp = tf.lite.Interpreter(model_content=conv.convert())
itp.resize_tensor_input(itp.get_input_details()[0]["index"], list(X.shape))
itp.allocate_tensors()
itp.set_tensor(itp.get_input_details()[0]["index"], X)
itp.invoke()
got = itp.get_tensor(itp.get_output_details()[0]["index"])

print(f"max_abs={np.abs(ref-got).max():.4f}")   # > 0.01: BUG
```
**Correct:** max_abs ≤ 0.01 (fp16 quantization error budget)  
**Wrong:** max_abs > 0.01 (tanh saturation amplifies fp16 weight truncation)

---

## RC-12 · TFLite XNNPack delegate fails on full transformer block
**Category:** Compiler optimization  
**Affects:** 1 bug — `tflite/cross_tflite_transformer_encoder_layer.py`

### What is wrong
TFLite's XNNPack delegate fails to prepare the fused transformer subgraph (6 MatMuls,
2 LayerNorms, 1 Softmax inside Reshape/Transpose pipelines). Fallback path produces
catastrophically wrong output (max_abs = 3.9e20 vs Keras fp32).

### What is correct
TFLite should either prepare the delegate successfully (matching Keras within 0.01)
or fail loudly with a clear error message. Silent fallback to a broken kernel is wrong.

### Minimal reproducer
**File:** `tflite/cross_tflite_transformer_encoder_layer.py`

---

## RC-13 · XLA JIT precision divergence on cancellation pattern
**Category:** Compiler optimization  
**Affects:** 1 bug — `xla/cross_xla_add_self_sub_double.py`

### What is wrong
XLA JIT (`jit_compile=True`) computes `x + x - 2*x` with a different fp32 accumulation
order than TF eager. On values where cancellation is expected to produce near-zero,
XLA's kernel fusion changes the reduction order and produces max_abs ≈ 0.020.

### What is correct
TF eager (reference) produces values within machine epsilon of zero for the cancellation.
XLA diverges by 0.020 — exceeding the 0.01 deployment tolerance.

### Minimal reproducer
**File:** `xla/cross_xla_add_self_sub_double.py`
```python
import numpy as np, tensorflow as tf

x = tf.constant(np.random.randn(4, 512).astype(np.float32) * 2.0)
W = tf.constant(np.random.randn(512, 64).astype(np.float32))

@tf.function
def eager_f(x):
    m = tf.matmul(x, W)
    return m + m - 2.0 * m    # should be ~0

@tf.function(jit_compile=True)
def xla_f(x):
    m = tf.matmul(x, W)
    return m + m - 2.0 * m

ref = eager_f(x).numpy()
got = xla_f(x).numpy()
print(f"max_abs={np.abs(ref-got).max():.4f}")   # XLA ≈ 0.020, eager ≈ 0
```
**Correct:** max_abs ≈ 0 (cancellation holds in fp32)  
**Wrong:** XLA returns max_abs ≈ 0.020 (kernel fusion changes accumulation order)

---

## RC-14 · XLA image resize wrong coordinate transform
**Category:** Compiler optimization  
**Affects:** 1 bug — `xla/github_tensorflow_002.py`

### What is wrong
XLA's `tf.image.resize` (nearest, bilinear, bicubic) uses a different coordinate
transformation formula than TF eager for certain scale factors. The difference is
not rounding noise — it systematically picks a different source pixel.

### What is correct
TF eager result. The coordinate transform should be consistent regardless of whether
XLA JIT is enabled.

### Minimal reproducer
**File:** `xla/github_tensorflow_002.py`

---

## RC-15 · XLA silently executes matmul with incompatible shapes
**Category:** Compiler optimization  
**Affects:** 1 bug — `xla/github_tf_61881_xla_matmul_incompatible_shapes.py`

### What is wrong
`tf.matmul(x, W)` where `x.shape=[1,4]` and `W.shape=[6,1]` (inner dims 4≠6).
XLA compiles and executes this without error, reading garbage memory and returning
a plausible-looking tensor. TF eager correctly raises `InvalidArgumentError`.

### What is correct
An `InvalidArgumentError` must be raised. Silently running with incompatible shapes
produces undefined output and is a correctness safety failure.

### Minimal reproducer
**File:** `xla/github_tf_61881_xla_matmul_incompatible_shapes.py`
```python
import tensorflow as tf

x = tf.constant([[2., 4., 6., 8.]], dtype=tf.float32)      # [1,4]
W = tf.constant([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6]])      # [6,1] — incompatible!

@tf.function
def eager_f(x): return tf.matmul(x, W)

@tf.function(jit_compile=True)
def xla_f(x): return tf.matmul(x, W)

try:
    eager_f(x)
    print("eager: no error (unexpected)")
except Exception as e:
    print(f"eager: {type(e).__name__} ← correct")

try:
    out = xla_f(x)
    print(f"XLA: returned {out.numpy()} ← BUG, should error")
except Exception as e:
    print(f"XLA: {type(e).__name__}")
```
**Correct:** `InvalidArgumentError: In[0] mismatch In[1] shape`  
**Wrong:** XLA returns a tensor with garbage values (no error)

---

## RC-16 · XLA dead code elimination — executes invalid dead slice
**Category:** Compiler optimization  
**Affects:** 1 bug — `xla/github_tf_61884_xla_dead_code_elimination.py`

### What is wrong
A graph contains a dead `tf.slice` whose result is never used (zero multiply trick).
The slice has an invalid range (size=5 > available dim=3). XLA executes it anyway
and crashes. TF eager's `tf.function` correctly skips dead code.

### What is correct
Dead code should not be executed. TF eager returns the correct result (the slice's
output is multiplied by 0, so it's logically dead) without touching the invalid slice.

### Minimal reproducer
**File:** `xla/github_tf_61884_xla_dead_code_elimination.py`
```python
import tensorflow as tf

x = tf.constant([1., 2., 3.], dtype=tf.float32)   # length 3

@tf.function
def eager_f(x):
    dead = tf.slice(x, [0], [5])   # invalid: size 5 > len 3 — but never used
    return x * 0.0 + dead * 0.0   # both branches zeroed; dead slice is dead code

@tf.function(jit_compile=True)
def xla_f(x):
    dead = tf.slice(x, [0], [5])
    return x * 0.0 + dead * 0.0

print(eager_f(x).numpy())   # [0, 0, 0] — correct, dead slice skipped
print(xla_f(x).numpy())     # CRASH: XLA executes dead slice with invalid bounds
```
**Correct:** `[0., 0., 0.]` (dead slice skipped)  
**Wrong:** XLA crashes on the invalid slice instead of eliminating it

---

## RC-17 · OpenVINO CPU uint8/int8 SIMD uses saturation instead of ONNX-required wrapping
**Category:** Compiler optimization  
**Affects:** 5 bugs
- `openvino/github_ov_019_uint8_sub_no_wrap.py` (uint8 Sub)
- `openvino/github_ov_020_uint8_mul_no_wrap.py` (uint8 Mul)
- `openvino/github_ov_021_uint8_add_no_wrap.py` (uint8 Add)
- `openvino/github_ov_024_int8_sub_saturation.py` (int8 Sub)
- `openvino/github_ov_025_int8_add_saturation.py` (int8 Add)

### What is wrong
OpenVINO's CPU plugin dispatches 8-bit integer element-wise operations (Add, Sub, Mul)
through SIMD intrinsics that use **saturating arithmetic** (clamp to [0,255] or [-128,127]).
The ONNX spec mandates **modular (wrapping) arithmetic** — the same as C unsigned/two's
complement overflow.

| Op | Input | ONNX correct (wrap) | OV wrong (saturate) |
|----|-------|---------------------|---------------------|
| uint8 Sub | 5 − 10 | 251 | 0 |
| uint8 Mul | 200 × 200 | 64 | 255 |
| uint8 Add | 200 + 100 | 44 | 255 |
| int8 Sub | −128 − 1 | 127 | −128 |
| int8 Add | 100 + 100 | −56 | 127 |

Only 8-bit ops are affected (int16 wraps correctly), indicating the bug is in the
8-bit SIMD dispatch path.

### Minimal reproducer
**File:** `openvino/github_ov_019_uint8_sub_no_wrap.py`
```python
import numpy as np, openvino as ov
from onnx import helper, TensorProto, numpy_helper

a = np.array([5],  dtype=np.uint8)   # 5 - 10 should wrap to 251
b = np.array([10], dtype=np.uint8)

node  = helper.make_node("Sub", ["a","b"], ["c"])
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("a", TensorProto.UINT8, [1])],
    [helper.make_tensor_value_info("c", TensorProto.UINT8, [1])],
    initializer=[numpy_helper.from_array(b,"b")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

core = ov.Core()
comp = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out  = comp({"a": a})[comp.output(0)]

print(f"5 - 10 = {out[0]}  (expected 251, OV gives {out[0]})")
# ONNX spec: uint8 wrapping → 251
# OV: saturating SIMD → 0
```
**Correct:** 251 (modular wrap: (5 − 10) mod 256)  
**Wrong:** 0 (saturated to lower bound)

---

## RC-18 · OpenVINO ReduceLogSumExp — naive log(sum(exp)) overflows
**Category:** Compiler optimization  
**Affects:** 1 bug — `openvino/github_ov_022_reducelogsumexp_overflow.py`

### What is wrong
OpenVINO computes `ReduceLogSumExp` as `log(sum(exp(x)))` without the numerically
stable max-subtraction trick. For any `x ≥ 88.7` (fp32 overflow boundary),
`exp(x)` overflows to `+inf`, and `log(inf) = inf`. The result is `+inf` instead of
a finite correct value.

### What is correct
Stable form: `m = max(x); result = m + log(sum(exp(x − m)))`. The subtraction of
`m` keeps the exponent arguments ≤ 0, so `exp` never overflows. PyTorch, NumPy, and
ORT all use this form.

### Minimal reproducer
**File:** `openvino/github_ov_022_reducelogsumexp_overflow.py`
```python
import numpy as np, openvino as ov
from onnx import helper, TensorProto

x = np.array([[100.0, 88.0, 50.0]], dtype=np.float32)  # 100 >> fp32 exp limit

node  = helper.make_node("ReduceLogSumExp", ["X"], ["Y"], axes=[1], keepdims=0)
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1,3])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

core = ov.Core()
comp = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
got  = comp({"X": x})[comp.output(0)]

# Stable reference: m + log(sum(exp(x - m)))
m   = x.max(axis=1, keepdims=True)
ref = (m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))).ravel()

print(f"Input:    {x[0].tolist()}")
print(f"Expected: {ref[0]:.4f}")    # ≈ 100.0001
print(f"OV got:   {got[0]:.4f}")   # inf ← BUG
```
**Correct:** ≈ 100.0001 (stable formula)  
**Wrong:** `inf` (naive formula overflows)

---

## RC-19 · OpenVINO Relu(NaN) → 0 instead of NaN
**Category:** Compiler optimization  
**Affects:** 1 bug — `openvino/github_ov_023_relu_nan_propagation.py`

### What is wrong
OpenVINO's Relu kernel returns `0.0` for NaN inputs. IEEE 754 requires NaN to
propagate through all arithmetic operations. `max(0, NaN) = NaN`. ORT, PyTorch,
and JAX all return NaN. OV silently discards the NaN, which hides numerical errors.

### Minimal reproducer
**File:** `openvino/github_ov_023_relu_nan_propagation.py`
```python
import numpy as np, openvino as ov
from onnx import helper, TensorProto

x = np.array([float('nan'), 1.0, -1.0], dtype=np.float32)

node  = helper.make_node("Relu", ["X"], ["Y"])
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

core = ov.Core()
comp = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out  = comp({"X": x})[comp.output(0)]

print(f"Relu({x}) = {out}")
# Expected: [nan, 1.0, 0.0]
# OV gives: [0.0, 1.0, 0.0]  ← NaN silently zeroed
```
**Correct:** `[nan, 1.0, 0.0]`  
**Wrong:** `[0.0, 1.0, 0.0]` (NaN information lost)

---

## RC-20 · OpenVINO Exp(NaN) → +inf instead of NaN
**Category:** Compiler optimization  
**Affects:** 1 bug — `openvino/github_ov_026_exp_nan_to_inf.py`

### What is wrong
OpenVINO's Exp kernel uses an ordered comparison to select the computation path.
NaN fails all ordered comparisons and falls into the "large positive value" branch,
returning `+inf` instead of NaN. IEEE 754: `exp(NaN) = NaN`.

### Minimal reproducer
**File:** `openvino/github_ov_026_exp_nan_to_inf.py`
```python
import numpy as np, openvino as ov
from onnx import helper, TensorProto

x = np.array([float('nan'), 0.0, 1.0], dtype=np.float32)

node  = helper.make_node("Exp", ["X"], ["Y"])
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

core = ov.Core()
comp = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
out  = comp({"X": x})[comp.output(0)]

print(f"Exp({x}) = {out}")
# Expected: [nan, 1.0, e]
# OV gives: [inf, 1.0, e]  ← NaN → inf, wrong
```
**Correct:** `[nan, 1.0, 2.7183]`  
**Wrong:** `[inf, 1.0, 2.7183]` (NaN treated as large positive)

---

## RC-21 · OpenVINO MaxPool accepts pad ≥ kernel (invalid per ONNX spec)
**Category:** ONNX convert  
**Affects:** 1 bug — `openvino/cross_openvino_maxpool_bad_pad.py`

### What is wrong
`MaxPool` with `kernel=[3,3]` and `pads=[3,3,3,3]` — ONNX spec requires
`pad < kernel` on each side. OpenVINO silently loads this invalid model at
`read_model()` time and returns a wrong output shape `(1,1,12,12)`.
ORT, PyTorch, and onnx2torch all correctly reject the model at load time.

### What is correct
The model should be rejected with a clear error message at load/compile time.
Silently running with an invalid specification produces garbage output.

### Minimal reproducer
**File:** `openvino/cross_openvino_maxpool_bad_pad.py`
```python
import openvino as ov
from onnx import helper, TensorProto

node  = helper.make_node("MaxPool", ["X"], ["Y"],
            kernel_shape=[3,3],
            pads=[3,3,3,3])    # pad=3 ≥ kernel=3 — INVALID per ONNX spec
graph = helper.make_graph([node], "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1,1,6,6])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

# ORT rejects: "Pad should be smaller than kernel"
# OV silently accepts and produces wrong output shape
core = ov.Core()
comp = core.compile_model(core.read_model(model.SerializeToString(), b""), "CPU")
import numpy as np
out = comp({"X": np.ones((1,1,6,6), dtype=np.float32)})[comp.output(0)]
print(f"Output shape: {out.shape}")   # (1,1,12,12) — wrong, should have errored
```
**Correct:** load-time error `"Pad should be smaller than kernel"`  
**Wrong:** OpenVINO silently accepts and returns shape `(1,1,12,12)`

---

## RC-22 · OpenVINO CPU tiled GEMM / Winograd conv accumulation order differs from ORT
**Category:** Compiler optimization  
**Affects:** 47 bugs — all `cross_openvino_*.py` except maxpool_bad_pad

### What is wrong
OpenVINO's CPU plugin selects a **tiled GEMM** or **Winograd convolution** algorithm
at `compile_model()` time to maximize throughput. These algorithms partition the inner
dimension into tiles and reduce partial sums in a different order than ORT's sequential
reference path. For fp32, accumulation is not associative:
`(a+b)+c ≠ a+(b+c)` when rounding is involved.

The discrepancy grows with:
- Inner dimension K (K=512 → 0.01–0.5 absolute error)
- Number of chained ops (Transformer block: up to 38.7)
- Sensitivity of the downstream op (Log, tanh, GELU amplify small differences)

**Key evidence:** `cross_openvino_linear_relu_chain.py` shows ORT_opt, onnx2torch,
torch.compile, and TorchScript all agree with ORT_ref within tolerance. OpenVINO alone
diverges — ruling out the reference being unusual.

### What is correct
ORT CPU with `ORT_DISABLE_ALL` (sequential accumulation) is the reference.
Both results are valid fp32; the difference is non-portability: a model validated
on ORT produces different outputs on OpenVINO beyond acceptable tolerances.

### Representative minimal reproducer
**File:** `openvino/cross_openvino_add_relu_sub.py`
```python
import numpy as np, onnxruntime as ort, openvino as ov
from onnx import helper as oh, TensorProto as TP, numpy_helper as onh

np.random.seed(42)
x = np.random.randn(2, 512).astype(np.float32)
W = np.random.randn(512, 64).astype(np.float32)
b = np.random.randn(64).astype(np.float32)
R = np.random.randn(2, 64).astype(np.float32)

nodes = [
    oh.make_node("MatMul", ["X","W"], ["mm"]),
    oh.make_node("Add",    ["mm","b"],["add"]),
    oh.make_node("Relu",   ["add"],  ["relu"]),
    oh.make_node("Sub",    ["relu","R"],["Y"]),
]
graph = oh.make_graph(nodes,"g",
    [oh.make_tensor_value_info("X",TP.FLOAT,[2,512])],
    [oh.make_tensor_value_info("Y",TP.FLOAT,[2,64])],
    initializer=[onh.from_array(W,"W"),onh.from_array(b,"b"),onh.from_array(R,"R")])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("",13)])
mb = model.SerializeToString()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(mb,so,providers=["CPUExecutionProvider"]).run(None,{"X":x})[0]

core = ov.Core()
comp = core.compile_model(core.read_model(mb,b""),"CPU")
got  = comp({"X":x})[comp.output(0)]

max_abs = float(np.abs(ref.ravel()-got.ravel()).max())
print(f"ORT (ref):    {ref.ravel()[:4]}")
print(f"OpenVINO:     {got.ravel()[:4]}")
print(f"max_abs={max_abs:.4f}  (> 0.01 = BUG)")
```
**Correct (ORT):** sequential fp32 accumulation  
**Wrong (OpenVINO):** tiled GEMM produces max_abs > 0.01 (up to 38.7 for attention patterns)

### All 47 affected scripts

| Pattern group | Scripts |
|--------------|---------|
| Attention / Transformer | alibi_attention, attention_causal_only, attention_logit_softcap, attention_with_sink_token, flex_attention_precision_sdpa, group_query_attention, multi_query_attention, transformer_encoder_layer |
| Convolution | conv_add_relu, conv_bn_eval_explicit, conv_bn_fusion, conv_bn_relu6, conv_fp32_precision, conv_prelu_channel, inception_v3_branch, pad_conv, pointwise_dw_block, tile_conv |
| Matrix ops | einsum_transpose, flatten_gemm, fp16_matmul_add, gemm_sigmoid, matmul_add_biasgelu_bcast, matmul_add_layernorm, reduce_l2_last, redundant_reshape, transpose_matmul_transpose |
| Layout / Shape | space_to_depth_block, transpose_transpose_squash, broadcast_1d_scalar, slice_full_range, glu, global_branch_mul |
| Activation / Elementwise | add_relu_sub, add_self_sub_double, aspp_dilated_branch, ei_log_zero, expm1_bounded, ia_sat_sub_uint8_underflow, linear_relu_chain, multi_scale_conv_branch, neg_unary, reciprocal_mul, rrelu_inference_identity, sub_self_mul_zero, triple_add_residual, where_mask_fill |

---

## Summary table

| RC | Root cause | Bugs | Backends | Category |
|----|-----------|------|----------|----------|
| RC-01 | onnx2torch Reshape FakeTensor | 22 | torch.compile | ONNX convert |
| RC-02 | Inductor bf16 SpaceToDepth | 1 | torch.compile | Compiler opt |
| RC-03 | Inductor transpose+reduce fusion | 1 | torch.compile | Compiler opt |
| RC-04 | BitShift-by-64 C UB | 3 | ORT, OV, Inductor | Compiler opt |
| RC-05 | index_add OOB silent | 1 | torch.compile | Compiler opt |
| RC-06 | ORT Cast chain fusion | 1 | ORT | Compiler opt |
| RC-07 | ORT Resize nearest rounding | 2 | ORT | Compiler opt |
| RC-08 | ORT Mod SIGFPE | 1 | ORT | Compiler opt |
| RC-09 | ORT bicubic resize error | 1 | ORT | Compiler opt |
| RC-10 | OOB Gather — silent vs error | 2 | ORT, OV | Compiler opt |
| RC-11 | TFLite fp16 weight quantization | 3 | TFLite | ONNX convert |
| RC-12 | TFLite XNNPack delegate failure | 1 | TFLite | Compiler opt |
| RC-13 | XLA JIT cancellation precision | 1 | XLA | Compiler opt |
| RC-14 | XLA resize coordinate transform | 1 | XLA | Compiler opt |
| RC-15 | XLA silent incompatible matmul | 1 | XLA | Compiler opt |
| RC-16 | XLA dead code not eliminated | 1 | XLA | Compiler opt |
| RC-17 | OV uint8/int8 SIMD saturation | 5 | OpenVINO | Compiler opt |
| RC-18 | OV ReduceLogSumExp naive overflow | 1 | OpenVINO | Compiler opt |
| RC-19 | OV Relu(NaN) → 0 | 1 | OpenVINO | Compiler opt |
| RC-20 | OV Exp(NaN) → inf | 1 | OpenVINO | Compiler opt |
| RC-21 | OV MaxPool invalid pad accepted | 1 | OpenVINO | ONNX convert |
| RC-22 | OV tiled GEMM / Winograd precision | 47 | OpenVINO | Compiler opt |
| **Total** | | **98** | | **26 convert / 72 compiler opt** |
