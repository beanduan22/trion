# Organized Real Bugs

All `.py` files in this directory are comment-free and use alternative inputs that trigger the same root cause class. Each script prints the buggy output together with the reference or expected correct output.

## `01_jax_masked_float64_attention_inf.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/01_jax_masked_float64_attention_inf.py
```

Output:

```text
all_false_small_shape:
  expected_sum=0.0
  bug=invalid value (inf) encountered in jit(convert_element_type). Because jax_config.debug_nans.value and/or config.jax_debug_infs is set, the de-optimized function (i.e., the function as if the `jit` decorator were removed) was called in an attempt to get a more precise error message. However, the de-optimized function did not produce invalid values during its execution. This behavior can result from `jit` optimizations causing the invalid value to be produced. It may also arise from having nan/inf constants as outputs, like `jax.jit(lambda ...: jax.numpy.nan)(...)`.
halfmask_tiny_shape:
  expected_sum=0.0
  bug=invalid value (inf) encountered in jit(convert_element_type). Because jax_config.debug_nans.value and/or config.jax_debug_infs is set, the de-optimized function (i.e., the function as if the `jit` decorator were removed) was called in an attempt to get a more precise error message. However, the de-optimized function did not produce invalid values during its execution. This behavior can result from `jit` optimizations causing the invalid value to be produced. It may also arise from having nan/inf constants as outputs, like `jax.jit(lambda ...: jax.numpy.nan)(...)`.
BUG REPRODUCED: masked float64 attention internal inf
```

## `02_tensorflow_nan_cast_cpu_gpu.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/02_tensorflow_nan_cast_cpu_gpu.py
```

Output:

```text
two_nans_float32_to_int32:
  ref=[-2147483648, -2147483648]
  cpu=[-2147483648, -2147483648]
  gpu=[0, 0]
nan_mixed_with_finite_float32_to_int32:
  ref=[-2147483648, 7, -3]
  cpu=[-2147483648, 7, -3]
  gpu=[0, 7, -3]
BUG REPRODUCED: CPU/GPU NaN cast inconsistency
```

## `03_tensorflow_sparse_negative_shape_crash.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/03_tensorflow_sparse_negative_shape_crash.py
```

Output:

```text
negative_dense_shape_more_cols:
  expected=ValueError: Invalid value in tensor used for shape: -1
  returncode=-6
  bug_tail=Status: INVALID_ARGUMENT: Expected shape dimensions to be non-negative, got -1
BUG REPRODUCED: SparseTensor negative dense_shape fatal crash
```

## `04_onnxruntime_cast_truncation.py`

Run:

```bash
python bugs/organized_real_bugs/04_onnxruntime_cast_truncation.py
```

Output:

```text
fractional_large_mix:
  x=[-3.9000000953674316, -0.4000000059604645, 0.4000000059604645, 3.9000000953674316]
  ort=[True, True, True, True]
  expected=[True, False, False, True]
tiny_signed_nonzeros:
  x=[-9.999999960041972e-13, -9.999999682655225e-21, 9.999999682655225e-21, 9.999999960041972e-13]
  ort=[True, True, True, True]
  expected=[False, False, False, False]
BUG REPRODUCED: float->int32->bool truncation bug
```

## `05_openvino_uint8_add_saturation.py`

Run:

```bash
python bugs/organized_real_bugs/05_openvino_uint8_add_saturation.py
```

Output:

```text
sparse_overflow_positions:
  a=[250, 10, 20, 30]
  b=[20, 20, 30, 40]
  ov=[255, 30, 50, 70]
  expected=[14, 30, 50, 70]
alternating_boundary:
  a=[255, 0, 255, 0]
  b=[1, 255, 2, 254]
  ov=[255, 255, 255, 254]
  expected=[0, 255, 1, 254]
BUG REPRODUCED: uint8 Add saturates instead of wrapping
```

## `06_torch_compile_runtime_reshape_crash.py`

Run:

```bash
python bugs/organized_real_bugs/06_torch_compile_runtime_reshape_crash.py
```

Output:

```text
shape_8x32:
  expected_sum=0.0
  eager_sum=0.0
  bug=InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')
shape_2x3x16:
  expected_sum=0.0
  eager_sum=0.0
  bug=InternalTorchDynamoError: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')
BUG REPRODUCED: FakeTensor runtime-reshape crash
```

## `07_tflite_fp16_associativity_break.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/07_tflite_fp16_associativity_break.py
```

Output:

```text
medium_dims_amp3:
  keras_head=[-0.016757139936089516, 0.05669211223721504, -0.011076530441641808, -0.04151429608464241]
  tflite_head=[-15.070952415466309, 13.735596656799316, 18.210630416870117, 10.150582313537598]
  max_abs=78.15414428710938
small_dims_amp8:
  keras_head=[-0.21681469678878784, 0.04495595023036003, 0.08843018114566803, -0.0819513127207756]
  tflite_head=[73.24822998046875, 11.073766708374023, 0.9743289947509766, 102.93243408203125]
  max_abs=337.00518798828125
BUG REPRODUCED: fp16 quantized associativity break
```

## `08_tvm_resize_halfpixel_semantics.py`

Run:

```bash
/home/binduan/miniconda3/envs/clawwork/bin/python bugs/organized_real_bugs/08_tvm_resize_halfpixel_semantics.py
```

Output:

```text
shape5_scale1.5_nearest_halfpixel:
  ort_head=[0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 0.0]
  tvm_head=[0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
  max_abs=6.0
shape5_scale1.25_nearest_halfpixel:
  ort_head=[0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  tvm_head=[0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  max_abs=6.0
BUG REPRODUCED: Resize half-pixel semantic mismatch
```

## `09_torch_compile_gh_158058_arange_dynamic_shapes.py`

Run:

```bash
python bugs/organized_real_bugs/09_torch_compile_gh_158058_arange_dynamic_shapes.py
```

Output:

```text
input: [3, 7]
expected: [3, 4, 5, 6]
bug_type: InternalTorchDynamoError
bug: PendingUnbackedSymbolNotFound: Pending unbacked symbols {u0, u1} not in returned outputs FakeTensor(..., size=(u0 - u1,), dtype=torch.int64) ((1,), 0).
BUG REPRODUCED: torch.compile arange dynamic-shape failure
```

## `10_torch_compile_gh_158561_copy_inplace_autograd.py`

Run:

```bash
python bugs/organized_real_bugs/10_torch_compile_gh_158561_copy_inplace_autograd.py
```

Output:

```text
input: [-1.0, 0.5, 3.0]
expected: [-3.0, 0.0, 5.0]
bug_type: BackendCompilerFailed
bug: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3, 1]], which is output 0 of ReluBackward0, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
BUG REPRODUCED: torch.compile copy_ inplace autograd failure
```

## `11_torch_compile_gh_158088_fake_tensor_intlist.py`

Run:

```bash
python bugs/organized_real_bugs/11_torch_compile_gh_158088_fake_tensor_intlist.py
```

Output:

```text
expected_shape: [640, 960]
expected_sum: 614400.0
bug_type: AssertionError
bug: Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'. Found in aten._local_scalar_dense.default(tensor(640, size=()))
BUG REPRODUCED: torch.compile FakeTensor intlist failure
```

## `12_torch_compile_gh_113793_no_grad_view_inplace.py`

Run:

```bash
python bugs/organized_real_bugs/12_torch_compile_gh_113793_no_grad_view_inplace.py
```

Output:

```text
input: [3.0, 4.0]
expected_after_call1: [5.0, 6.0]
expected_after_call2: [7.0, 8.0]
bug_type: BackendCompilerFailed
bug: RuntimeError: A view was created in no_grad mode and its base or another view of its base has been modified inplace with grad mode enabled. Given that this use case is ambiguous and error-prone, it is forbidden. You can clarify your code by moving both the view and the inplace either both inside the no_grad block (if you don't want the inplace to be tracked) or both outside (if you want the inplace to be tracked).
BUG REPRODUCED: torch.compile no_grad view inplace failure
```

## `13_openvino_gh_23088_maxpool_bad_pad.py`

Run:

```bash
python bugs/organized_real_bugs/13_openvino_gh_23088_maxpool_bad_pad.py
```

Output:

```text
kernel4_pad4_square:
  expected=ORT reject invalid pads>=kernel
  ort=[ONNXRuntimeError] : 1 : FAIL : Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
  openvino=[1, 1, 12, 12]
kernel3_pad_topbottom_only:
  expected=ORT reject invalid pads>=kernel
  ort=[ONNXRuntimeError] : 1 : FAIL : Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cpu/nn/pool_attributes.h:78 onnxruntime::PoolAttributes::PoolAttributes(const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>&, const std::string&, int) pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim] was false. Pad should be smaller than kernel.
  openvino=[1, 1, 12, 6]
BUG REPRODUCED: OpenVINO silently accepts invalid MaxPool padding
```

## `14_openvino_gh_33518_uint8_sub_wrap.py`

Run:

```bash
python bugs/organized_real_bugs/14_openvino_gh_33518_uint8_sub_wrap.py
```

Output:

```text
mixed_underflow_pattern:
  a=[1, 128, 255, 40]
  b=[2, 200, 1, 90]
  expected=[255, 184, 254, 206]
  openvino=[0, 0, 254, 0]
alternating_edges:
  a=[0, 10, 250, 3]
  b=[255, 11, 251, 200]
  expected=[1, 255, 255, 59]
  openvino=[0, 0, 0, 0]
BUG REPRODUCED: OpenVINO uint8 Sub saturates instead of wrapping
```

## `15_tvm_gh_gelu_approx_tanh.py`

Run:

```bash
/home/binduan/miniconda3/envs/clawwork/bin/python bugs/organized_real_bugs/15_tvm_gh_gelu_approx_tanh.py
```

Output:

```text
expected_control_max_abs: 2.384185791015625e-07
expected_tanh_head: [-7.009506225585938e-05, -0.015084296464920044, -0.13228577375411987, -0.1542859971523285]
tvm_tanh_head: [-0.00012671947479248047, -0.015524178743362427, -0.13206221163272858, -0.1542687714099884]
tanh_max_abs: 0.0004398822784423828
BUG REPRODUCED: TVM ignores Gelu approximate='tanh'
```

## `16_onnxruntime_gh_25264_cubic_resize_cpu.py`

Run:

```bash
python bugs/organized_real_bugs/16_onnxruntime_gh_25264_cubic_resize_cpu.py
```

Output:

```text
expected_head: [0.1286475658416748, 0.7080503106117249, 0.3708721399307251, 0.38750749826431274, 0.7414761781692505, 0.7187420725822449, 0.6016472578048706, 0.4305752217769623]
ort_head: [0.16084957122802734, 0.6892130970954895, 0.40063977241516113, 0.37476134300231934, 0.7290555834770203, 0.7193203568458557, 0.5845029950141907, 0.43745243549346924]
max_abs: 0.05667392909526825
BUG REPRODUCED: ONNX Runtime CPU cubic resize diverges from PyTorch bicubic
```

## `17_tensorflow_gh_106175_nan_cast_cpu_gpu.py`

Opened:

```text
December 13, 2025
Status: Open
```

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/17_tensorflow_gh_106175_nan_cast_cpu_gpu.py
```

Output:

```text
tf_cast_nan_pair:
  expected=[-2147483648, -2147483648]
  cpu=[-2147483648, -2147483648]
  gpu=[0, 0]
tf_numpy_array_nan_mix:
  expected=[-2147483648, 5, -2]
  cpu=[-2147483648, 5, -2]
  gpu=[0, 5, -2]
BUG REPRODUCED: TensorFlow CPU/GPU NaN cast inconsistency across APIs
```

## `18_tflite_gh_105833_default_opt_precision_loss.py`

Opened:

```text
December 8, 2025
Status: Open
```

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/18_tflite_gh_105833_default_opt_precision_loss.py
```

Output:

```text
expected_no_opt_mae: 0.0
optimized_mae: 0.009280232712626457
ratio: inf
tf_head: [-2.3844692707061768, 0.3198031485080719, 1.5558490753173828, -0.9517636299133301]
tflite_head: [-2.396664619445801, 0.32423722743988037, 1.570229172706604, -0.9461376070976257]
BUG REPRODUCED: TFLite DEFAULT optimization causes severe precision loss
```

## `19_openvino_gh_32839_reducelogsumexp_overflow.py`

Opened:

```text
November 13, 2025
Status: Open
```

Run:

```bash
python bugs/organized_real_bugs/19_openvino_gh_32839_reducelogsumexp_overflow.py
```

Output:

```text
input: [[89.5, 89.0, 0.0, -2.0], [120.0, 110.0, -5.0, -9.0]]
expected: [89.97407531738281, 120.00004577636719]
openvino: [inf, inf]
has_inf: True
BUG REPRODUCED: OpenVINO ReduceLogSumExp overflows without stable logsumexp
```

## `20_tensorflow_xla_gh_105654_variable_creation_in_call.py`

Opened:

```text
December 4, 2025
Status: Open
```

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/20_tensorflow_xla_gh_105654_variable_creation_in_call.py
```

Output:

```text
eager_shape: (4, 1, 1)
expected_compiled=shape like eager
bug_type: ValueError
bug: tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function. See https://www.tensorflow.org/guide/function#creating_tfvariables for more information.
BUG REPRODUCED: TensorFlow XLA variable creation inside call fails under jit_compile
```

## `21_tensorflow_xla_gh_105728_nested_tffunction_layer.py`

Opened:

```text
December 5, 2025
Status: Open
```

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/21_tensorflow_xla_gh_105728_nested_tffunction_layer.py
```

Output:

```text
eager_shape: (6, 4)
child_bug_type: NotImplementedError
child_bug: Exception encountered when calling MemoryNetwork.call().
expected_compiled=shape like eager
bug_type: NotImplementedError
bug: Exception encountered when calling MemoryNetwork.call().
BUG REPRODUCED: TensorFlow XLA nested tf.function inside Layer.call fails
```

## `22_tensorflow_xla_gh_105652_numpy_function_unknown_shape.py`

Opened:

```text
December 4, 2025
Status: Open
```

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/22_tensorflow_xla_gh_105652_numpy_function_unknown_shape.py
```

Output:

```text
eager_shape: (3, 8)
child_bug_type: ValueError
child_bug: Exception encountered when calling TestModel.call().
expected_compiled=shape like eager
bug_type: ValueError
bug: Exception encountered when calling TestModel.call().
BUG REPRODUCED: TensorFlow XLA tf.numpy_function leaves unknown TensorShape
```

## `23_tensorflow_gh_105829_cpu_conv2d_vs_pytorch.py`

Opened:

```text
December 8, 2025
Status: Open
```

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/23_tensorflow_gh_105829_cpu_conv2d_vs_pytorch.py
```

Output:

```text
pt_shape: (1, 64, 9, 9)
tf_shape: (1, 64, 9, 9)
max_abs: 0.0001373291015625
mean_abs: 1.0329834367439616e-05
mismatch_count: 1949
pt_head: [15.21281623840332, 30.195964813232422, 8.94794750213623, -17.01412582397461, -25.379106521606445, -31.354833602905273]
tf_head: [15.212820053100586, 30.19594955444336, 8.94793701171875, -17.014135360717773, -25.379098892211914, -31.35480308532715]
BUG REPRODUCED: TensorFlow CPU Conv2D diverges from PyTorch baseline
```

## `24_tflite_gh_105833_default_opt_conv2d_precision_loss.py`

Opened:

```text
December 8, 2025
Status: Open
```

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/24_tflite_gh_105833_default_opt_conv2d_precision_loss.py
```

Output:

```text
expected_no_opt_mae: 0.0
optimized_mae: 0.007029848173260689
optimized_max_abs: 0.03974698483943939
ratio: inf
tf_head: [0.5282237529754639, -0.3180733025074005, 1.4130181074142456, -0.28968122601509094, 1.2506710290908813, -0.11967551708221436]
tflite_head: [0.5340241193771362, -0.3158123791217804, 1.4132509231567383, -0.2981828451156616, 1.2534921169281006, -0.12119942903518677]
BUG REPRODUCED: TFLite DEFAULT optimization causes Conv2D precision loss
```

## `25_openvino_gh_023_relu_nan_propagation.py`

Run:

```bash
python bugs/organized_real_bugs/25_openvino_gh_023_relu_nan_propagation.py
```

Output:

```text
input: [nan, -1.0, 0.0, 1.0, inf]
expected: [nan, 0.0, 0.0, 1.0, inf]
openvino: [0.0, 0.0, 0.0, 1.0, inf]
expected_nan_at_0: True
openvino_nan_at_0: False
BUG REPRODUCED: OpenVINO Relu turns NaN into 0 instead of propagating NaN
```

## `26_openvino_gh_026_exp_nan_to_inf.py`

Run:

```bash
python bugs/organized_real_bugs/26_openvino_gh_026_exp_nan_to_inf.py
```

Output:

```text
input: [nan, 1.0]
expected: [nan, 2.7182817459106445]
openvino: [inf, 2.7182819843292236]
expected_nan_at_0: True
openvino_nan_at_0: False
openvino_inf_at_0: True
BUG REPRODUCED: OpenVINO Exp does not propagate NaN
```

## `27_onnxruntime_mod_zero_divisor_sigfpe.py`

Run:

```bash
python bugs/organized_real_bugs/27_onnxruntime_mod_zero_divisor_sigfpe.py
```

Output:

```text
all_zero_divisors:
  a= [7, 13]
  b= [0, 0]
  expected=safe exception or defined handling, never process-killing signal
  returncode=-8
  signal=SIGFPE
mixed_sign_zero_divisors:
  a= [-9, 5]
  b= [0, 0]
  expected=safe exception or defined handling, never process-killing signal
  returncode=-8
  signal=SIGFPE
BUG REPRODUCED: ONNXRuntime Mod with zero divisor kills the process with SIGFPE
```

## `28_tensorflow_xla_gh_61881_invalid_matmul_shape_check.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/28_tensorflow_xla_gh_61881_invalid_matmul_shape_check.py
```

Output:

```text
shape_1x4_by_6x1:
  input_shape=(1, 4)
  weight_shape=(6, 1)
  expected_error=InvalidArgumentError: Exception encountered when calling EagerModel.call().
  xla_output=[[42.000003814697266]]
  xla_error=None
BUG REPRODUCED: TensorFlow XLA executes invalid matmul shapes that eager rejects
```

## `29_jax_xla_add_self_sub_double.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/29_jax_xla_add_self_sub_double.py
```

Output:

```text
expected_head: [-2.4493327140808105, 4.787549018859863, 2.163334846496582, 38.809417724609375]
xla_head: [-2.451444625854492, 4.786455154418945, 2.1704235076904297, 38.822906494140625]
max_abs: 0.020000457763671875
BUG REPRODUCED: JAX/XLA add_self_sub_double numerically drifts from ONNX Runtime
```

## `30_tvm_foldconstant_inf_minus_inf.py`

Run:

```bash
/home/binduan/miniconda3/envs/clawwork/bin/python bugs/organized_real_bugs/30_tvm_foldconstant_inf_minus_inf.py
```

Output:

```text
input: [1.0, 2.0, 3.0, 4.0]
expected: [nan, nan, nan, nan]
tvm: [0.0, 0.0, 0.0, 0.0]
max_abs: 1.0
BUG REPRODUCED: TVM FoldConstant rewrites inf*X - inf*X to zeros instead of NaNs
```

## `31_tensorflow_xla_gh_61884_dead_slice_not_eliminated.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/31_tensorflow_xla_gh_61884_dead_slice_not_eliminated.py
```

Output:

```text
input_shape: (1, 2, 3, 2)
expected_shape: (1, 2, 1, 2)
expected_error: None
xla_shape: None
xla_error: InvalidArgumentError: Exception encountered when calling XlaModel.call().
BUG REPRODUCED: TensorFlow XLA executes a dead invalid slice instead of eliminating it
```

## `32_tflite_transformer_xnnpack_prepare.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/32_tflite_transformer_xnnpack_prepare.py
```

Output:

```text
expected_head: [0.6222167611122131, 0.9313083291053772, 1.0119620561599731, 1.4583815336227417]
tflite_error: RuntimeError: tensorflow/lite/kernels/read_variable.cc:67 variable != nullptr was not true.Node number 4 (READ_VARIABLE) failed to invoke.
BUG REPRODUCED: TFLite XNNPack prepare fails while Keras runs correctly
```

## `33_torch_compile_gh_121135_index_add_shape_validation.py`

Run:

```bash
python bugs/organized_real_bugs/33_torch_compile_gh_121135_index_add_shape_validation.py
```

Output:

```text
case=(32, 4)+(4,)
  eager_error: RuntimeError: source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [32, 4] source.shape = [4]
  compiled_shape: [32, 4]
  compiled_error: None
  torchscript_error: RuntimeError: source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [32, 4] source.shape = [32]
case=(8, 3)+(3,)
  eager_error: RuntimeError: source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [8, 3] source.shape = [3]
  compiled_shape: [8, 3]
  compiled_error: None
  torchscript_error: RuntimeError: source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [8, 3] source.shape = [8]
BUG REPRODUCED: torch.compile skips index_add shape validation that eager and TorchScript enforce
```

## `34_onnxruntime_resize_nearest_pixel_off.py`

Run:

```bash
python bugs/organized_real_bugs/34_onnxruntime_resize_nearest_pixel_off.py
```

Output:

```text
expected_slice_30_35: [12.0, 12.0, 13.0, 13.0, 13.0]
ort_slice_30_35: [12.0, 12.0, 13.0, 13.0, 14.0]
mismatch_count: 12
BUG REPRODUCED: ONNX Runtime nearest resize picks one-pixel-off source positions
```

## `35_onnxruntime_resize_round_prefer_ceil_tie.py`

Run:

```bash
python bugs/organized_real_bugs/35_onnxruntime_resize_round_prefer_ceil_tie.py
```

Output:

```text
expected: [0.05263157933950424, 0.21052631735801697, 0.42105263471603394, 0.5789473652839661, 0.7894737124443054, 0.9473684430122375]
ort: [0.05263157933950424, 0.2631579041481018, 0.42105263471603394, 0.5789473652839661, 0.7368420958518982, 0.9473684430122375]
expected_elem4: 0.7894737124443054
ort_elem4: 0.7368420958518982
BUG REPRODUCED: ONNX Runtime mis-rounds half-pixel nearest resize ties under round_prefer_ceil
```

## `36_tflite_attention_logit_softcap.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/36_tflite_attention_logit_softcap.py
```

Output:

```text
expected_head: [2.414766311645508, 3.952533721923828, 0.11160194128751755, 3.2639355659484863]
tflite_head: [2.4152097702026367, 3.953533887863159, 0.11113925278186798, 3.264158248901367]
max_abs: 0.027167201042175293
BUG REPRODUCED: TFLite attention_logit_softcap diverges after fp16 weight quantization
```

## `37_tensorflow_xla_resize_semantics_drift.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/37_tensorflow_xla_resize_semantics_drift.py
```

Output:

```text
nearest_3x3_to_5x5:
  eager_head: [1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0]
  xla_head: [1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0]
  max_abs: 0.0
bilinear_antialias_8x8_to_3x3:
  eager_head: [0.5139087438583374, 0.5161528587341309, 0.2832993268966675, -0.4555318355560303, 0.2234289050102234, 0.18980616331100464, -0.6522951722145081, -0.2516065835952759]
  xla_error: InvalidArgumentError: Detected unsupported operations when trying to compile graph __inference_f_48[_XlaMustCompile=true,config_proto=13561319589895757934,executor_type=11160318154034397263] on XLA_CPU_JIT: ScaleAndTranslate (No registered 'ScaleAndTranslate' OpKernel for XLA_CPU_JIT devices compatible with node {{node resize/ScaleAndTranslate}}){{node resize/ScaleAndTranslate}}
bicubic_5x5_to_8x8:
  eager_head: [0.23463737964630127, -0.013777382671833038, -0.5524725317955017, -1.5928817987442017, -1.2163257598876953, 0.3673795759677887, -0.2620570659637451, -0.9771168828010559]
  xla_error: InvalidArgumentError: Detected unsupported operations when trying to compile graph __inference_f_61[_XlaMustCompile=true,config_proto=13561319589895757934,executor_type=11160318154034397263] on XLA_CPU_JIT: ResizeBicubic (No registered 'ResizeBicubic' OpKernel for XLA_CPU_JIT devices compatible with node {{node resize/ResizeBicubic}}){{node resize/ResizeBicubic}}
BUG REPRODUCED: TensorFlow XLA resize behavior diverges from eager or fails to compile eager-supported cases
```

## `38_torch_compile_redundant_reshape_crash.py`

Run:

```bash
python bugs/organized_real_bugs/38_torch_compile_redundant_reshape_crash.py
```

Output:

```text
expected_head: [-2.705871343612671, 3.6985130310058594, 1.053196907043457, 41.27541732788086]
eager_head: [-2.705871105194092, 3.6985108852386475, 1.0531883239746094, 41.27541732788086]
bug_type: InternalTorchDynamoError
bug: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')
BUG REPRODUCED: torch.compile crashes on redundant reshape patterns after ONNX conversion
```

## `39_onnxruntime_bitshift_shift64_ub.py`

Run:

```bash
python bugs/organized_real_bugs/39_onnxruntime_bitshift_shift64_ub.py
```

Output:

```text
input_values: [1000, 255]
shift_values: [64, 64]
expected: [0, 0]
ort: [1000, 255]
BUG REPRODUCED: ONNX Runtime BitShift by 64 returns the original value instead of zero
```

## `40_openvino_int8_sub_saturation.py`

Run:

```bash
python bugs/organized_real_bugs/40_openvino_int8_sub_saturation.py
```

Output:

```text
a: [-128, 127, -1, 0]
b: [1, -128, -127, 100]
expected: [127, -1, 126, -100]
openvino: [-128, 127, 126, -100]
BUG REPRODUCED: OpenVINO int8 Sub saturates instead of two's-complement wrapping
```

## `41_openvino_int8_add_saturation.py`

Run:

```bash
python bugs/organized_real_bugs/41_openvino_int8_add_saturation.py
```

Output:

```text
a: [100, 127, -128, -100]
b: [100, 10, -10, -50]
expected: [-56, -119, 118, 106]
openvino: [127, 127, -128, -128]
BUG REPRODUCED: OpenVINO int8 Add saturates instead of two's-complement wrapping
```

## `42_openvino_oob_gather_returns_zero.py`

Run:

```bash
python bugs/organized_real_bugs/42_openvino_oob_gather_returns_zero.py
```

Output:

```text
params: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
indices: [0, 3, 6, 9, 12, 1]
expected_error: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gather node. Name:'' Status Message: indices element out of data bounds, idx=12 must be within the inclusive range [-12,11]
openvino: [1.0, 4.0, 7.0, 10.0, 0.0, 2.0]
BUG REPRODUCED: OpenVINO silently returns 0 for an out-of-bounds Gather index
```

## `43_openvino_uint8_mul_saturation.py`

Run:

```bash
python bugs/organized_real_bugs/43_openvino_uint8_mul_saturation.py
```

Output:

```text
a: [200, 100, 50, 10]
b: [200, 3, 6, 30]
expected: [64, 44, 44, 44]
openvino: [255, 255, 255, 255]
BUG REPRODUCED: OpenVINO uint8 Mul saturates instead of modular wrapping
```

## `44_openvino_conv_fp32_precision.py`

Run:

```bash
python bugs/organized_real_bugs/44_openvino_conv_fp32_precision.py
```

Output:

```text
expected_head: [4.266868591308594, 9.87952709197998, 4.723992347717285, -1.4073364734649658, 2.5287177562713623, -2.3463897705078125, -5.512152194976807, -0.4095015525817871]
openvino_head: [4.272923469543457, 9.873534202575684, 4.731034755706787, -1.4000506401062012, 2.528006076812744, -2.3329763412475586, -5.502569198608398, -0.4092235565185547]
max_abs: 0.053732872009277344
BUG REPRODUCED: OpenVINO float32 Conv diverges from ONNX Runtime beyond tolerance
```

## `45_tflite_sub_self_mul_zero.py`

Run:

```bash
/home/binduan/miniconda3/envs/xcomp_gpu/bin/python bugs/organized_real_bugs/45_tflite_sub_self_mul_zero.py
```

Output:

```text
expected_head: [1.499847173690796, -0.2011507749557495, 0.6442595720291138, 0.6500227451324463]
tflite_head: [1530.1123046875, -852.1825561523438, -504.11700439453125, 42.33075714111328]
max_abs: 2250.31884765625
tflite_abs_max: 2250.28076171875
BUG REPRODUCED: TFLite breaks associativity in sub-self-mul-zero style matmul chains
```

## `46_torch_compile_topk_last_axis_k1.py`

Run:

```bash
python bugs/organized_real_bugs/46_torch_compile_topk_last_axis_k1.py
```

Output:

```text
expected: [1.7554446458816528, 2.568927049636841, 1.2612801790237427, 0.7792879939079285]
eager: [1.7554446458816528, 2.568927049636841, 1.2612801790237427, 0.7792879939079285]
bug_type: InternalTorchDynamoError
bug: TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'FakeTensor')
BUG REPRODUCED: torch.compile crashes on TopK followed by reshape
```

## `47_openvino_add_relu_sub_drift.py`

Run:

```bash
python bugs/organized_real_bugs/47_openvino_add_relu_sub_drift.py
```

Output:

```text
expected_head: [0.15072493255138397, 3.4932332038879395, 0.6986246109008789, 37.59840774536133]
openvino_head: [0.15072493255138397, 3.456390857696533, 0.6670341491699219, 37.61678695678711]
max_abs: 0.12434864044189453
BUG REPRODUCED: OpenVINO add-relu-sub chain drifts beyond tolerance from ORT
```

## `48_openvino_reciprocal_mul_drift.py`

Run:

```bash
python bugs/organized_real_bugs/48_openvino_reciprocal_mul_drift.py
```

Output:

```text
expected_head: [-1537.535888671875, -450.9237060546875, -297.34832763671875, 101.56495666503906]
openvino_head: [-1533.70703125, -449.8779296875, -296.828125, 98.7041015625]
max_abs: 9.35009765625
BUG REPRODUCED: OpenVINO reciprocal-matmul chain diverges massively from ORT
```
