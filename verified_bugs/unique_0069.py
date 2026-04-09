#!/usr/bin/env python3
"""
Bug #0069: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['fusion', 'matmul_bias_relu'], ['constant', 'relu_residual_sum'], ['normalization', 'reduce_l2_last'], ['layout', 'gather_slice_combo'], ['branch', 'add_layernorm'], ['fusion', 'gemm_alpha_beta_transb']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0069.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0069.onnx")
INPUT = np.frombuffer(bytes.fromhex("d975d4be18eee3be7d31d8bf272474bf44e65b3f6ff39fbf046a493f15631c3fbea1c33dbdee91bf18f3a53d0dac7abfc10a5a3fe9fab13ea2c91f3fee7129bfb52b8d3e66f79e3eeca54f3efabc40beb9048cbd966681bf3463f7bd2ac6d7bf52d509bf4bbcc1bfdf6c8abffcad753f99682b3ff04d91bfb24e273f26084fbe2964ef3ecbd4b9bee4a8b73e77c3c33e0adbd9be47cb64bf7dcd86bfc210063e7e89b23f30ac383e15aa98be4679323eae9144bed25abebf0415d1bfa4c9a33d841686bf3c36c7be81c1913ebb3ee4bf8cc24dbf1816e83f2594bc3e97583c3f8b15053f8c993d3efa3c88bfa5ec9abe8d0a65bf122df53f4d8fbf3fba8ce4be16e3d73e755f0abeae96b43f58fc9ebf98a60cbf0c2922be0fcf9dbe76332e3e5487b4be9094933e210e88bfff576dbf7a1bbe3cfebbbdbd9b0de13f15cdfbbf37e096bf3dfec33e5899123e379251bf949429bfb09b7e3ee45d8fbffe96183fbe74943fe6631a3f1ffeb0be8c7f14bf5245033e8f71d2be65fd333f5845a7beb3882bbf0a5fa63eb231fd3fdca0623f4a61d1be93020e3f5097e43efa71a9bf837a81bffc2cfebff5d3083de2ac8b3e4c74b1bf29f316407457bbbfafd2a03eb011853e3c1e53bfc41238bfb4a7813f95c388be24e7b8bf81890fc048d483be00de93bfe0e094bf1c14f3bf2470023e2cd2a2bf5aea5bbf773cab3f5b3c1b3c0f38483ff91b2abf819330bf7a5965bf49a850bec0134ebefc2e403d6f4da8bf4191873ee7ad10bf7dc169bf7f18d6bf92ff983f06cf24c044eab43f343bd8be068a083f86e9c13f8442d43f7000123f7e6e8ebc84a08cbec7ce87bffbbd063fd14fcabfa653f0be71a4a1bf9f339fbfde4cfdbe0197d63e9a2ed3beb71e75bfd3f58d3edf48623f495c22bece05a53f42978fbfed3c65bfedacb6be4840573eb8733f3e4131873fcb1b643ff026423c9ffbcd3ef4e2343f8529b8bf2a4798bc1b9838c0ba216c3d75e4ae3f4446e53e1edac03d94437cbee88536bf488c1b3f024b90bfe4b547be3fc6a83f1ac291be5942693f095179bf673300bffd7e34be5ae38abf392de33ff5bd9cbd862bf93e8af2bbbe4c6280be090f953fef6580bfbab2bcbfed9265bfd85d40bf1edf05409536aabd5705b23ecab3693fd44e9bbf1a1aadbfa400bcbe0e1c433f71bd243e6a3ec2bf05559d3ecea80cbf334f35bf6132b7bff3ffffbd4600af3f0a20143f3b5fa4bebfee723e6eaf5fbf56321c3f047b2b3f4c4857bf32dc37bffc2ecfbf1faac93fdce25abcb153b53f9a06fa3f2d2d573fc82b783fb2b018bf6d736f3f48d84e3e0b7c4bc0ed117e3f76be173e24363e3f9be305c07c09f63f7f16aa3f97ba2fbfc1d3733fedef063f2d93d53f5c55343fdeae9cbe6547b5be7e24ba3e6a9a93bfc664a43e4e5243bd61f188bf"), dtype=np.float32).reshape([1, 256])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
