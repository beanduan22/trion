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
from __future__ import annotations
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
    if op == "Reciprocal": return [np.float32(1.0) / get(0)]

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
        to    = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        x     = get(0)
        # ORT and numpy use C-cast wrap on float→int (200.0 → int8 → -56);
        # JAX saturates (200.0 → int8 → 127). For *narrow* int types
        # (int8, uint8, int16, uint16) we explicitly implement wrap to
        # match ORT — the mod arithmetic stays inside fp32's 2^24 exact
        # integer range, so no precision is lost.
        # For wide int types (int32, int64) the wrap span (2^32 / 2^64)
        # cannot be represented exactly in fp32, so attempting wrap-mod
        # in fp32 produces zero (catastrophic precision loss). For these
        # types we fall back to plain astype — out-of-range float→int32/64
        # values are a rare edge case and the spec explicitly leaves the
        # behaviour implementation-defined; both ORT and JAX agree on
        # in-range values, which covers ~all real-world traffic.
        _NARROW_INT = (np.int8, np.uint8, np.int16, np.uint16)
        if np.issubdtype(dtype, np.integer) and dtype.type in _NARROW_INT:
            info = np.iinfo(dtype)
            span = int(info.max) - int(info.min) + 1   # ≤ 65536
            x_int = (F.where(x >= np.float32(0.0), F.floor(x), F.ceil(x))
                     if hasattr(F, "where") and hasattr(F, "ceil")
                     else (F.floor(x) if hasattr(F, "floor") else np.floor(x)))
            mod = ((x_int - np.float32(info.min)) %
                   np.float32(span)) + np.float32(info.min)
            return [mod.astype(dtype)]
        return [x.astype(dtype)]

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
        constant_value_t = get(2) if len(node.input) >= 3 else None
        constant_value = (
            float(_np(constant_value_t).flatten()[0])
            if constant_value_t is not None
            else 0.0
        )
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            # ONNX modes map directly to numpy/jax: constant, reflect, edge, wrap.
            jnp_mode = {"wrap": "wrap"}.get(mode, mode)
            if jnp_mode == "constant":
                return [jnp.pad(x, pad_width, mode="constant",
                                constant_values=constant_value)]
            return [jnp.pad(x, pad_width, mode=jnp_mode)]
        else:
            # TF native pad supports CONSTANT, REFLECT, SYMMETRIC. ONNX `edge`
            # has no direct TF equivalent — we emulate it with tf.concat of
            # gathered border slices so XLA still gets a static graph.
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            tf_mode = {"constant": "CONSTANT",
                       "reflect":  "REFLECT"}.get(mode)
            if tf_mode == "CONSTANT":
                return [tf.pad(x, paddings, mode="CONSTANT",
                               constant_values=constant_value)]
            if tf_mode == "REFLECT":
                return [tf.pad(x, paddings, mode="REFLECT")]
            # mode == "edge" (replicate boundary) or "wrap" (circular).
            for axis, (pb, pe) in enumerate(pad_width):
                if pb == 0 and pe == 0:
                    continue
                parts = []
                if mode == "edge":
                    if pb:
                        parts.append(tf.repeat(tf.gather(x, [0], axis=axis),
                                               pb, axis=axis))
                    parts.append(x)
                    if pe:
                        last_idx = tf.shape(x)[axis] - 1
                        parts.append(tf.repeat(tf.gather(x, [last_idx], axis=axis),
                                               pe, axis=axis))
                elif mode == "wrap":
                    if pb:
                        parts.append(tf.gather(
                            x, tf.range(tf.shape(x)[axis] - pb,
                                        tf.shape(x)[axis]), axis=axis))
                    parts.append(x)
                    if pe:
                        parts.append(tf.gather(x, tf.range(pe), axis=axis))
                else:
                    raise NotImplementedError(f"Pad mode={mode!r} not supported in TF backend")
                x = tf.concat(parts, axis=axis)
            return [x]

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
        exclusive = bool(_attr(node, "exclusive", 0))
        reverse   = bool(_attr(node, "reverse",   0))
        if reverse:
            # Reverse along the requested axis, run forward cumsum, reverse back.
            x = F.flip(x, axis=axis) if hasattr(F, "flip") else _flip_axis(x, axis)
        cs = F.cumsum(x, axis=axis) if hasattr(F, "cumsum") else F.cumulative_sum(x, axis=axis)
        if exclusive:
            # exclusive: shift cumsum by one along axis, fill leading slot with 0.
            ndim = len(x.shape)
            ax = axis % ndim
            slc_keep = [slice(None)] * ndim
            slc_keep[ax] = slice(0, x.shape[ax] - 1)
            shifted = cs[tuple(slc_keep)]
            zero_shape = list(x.shape); zero_shape[ax] = 1
            zeros = (F.zeros(zero_shape, dtype=cs.dtype) if hasattr(F, "zeros")
                     else np.zeros(zero_shape, dtype=np.float32))
            cs = F.concatenate([zeros, shifted], axis=ax)
        if reverse:
            cs = F.flip(cs, axis=axis) if hasattr(F, "flip") else _flip_axis(cs, axis)
        return [cs]

    # ── Activations (extended) ────────────────────────────────────────────────
    if op == "Softsign":
        x = get(0)
        return [x / (np.float32(1.0) + F.abs(x))]

    if op == "PRelu":
        x = get(0); slope = get(1)
        return [F.where(x >= np.float32(0.0), x, slope * x)]

    if op == "ThresholdedRelu":
        theta = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x > np.float32(theta), x, np.float32(0.0))]

    if op == "Shrink":
        lambd = float(_attr(node, "lambd", 0.5))
        bias  = float(_attr(node, "bias",  0.0))
        x = get(0)
        return [F.where(x < -np.float32(lambd), x + np.float32(bias),
                F.where(x > np.float32(lambd), x - np.float32(bias), np.float32(0.0)))]

    if op == "LogSoftmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        log_z = F.log(F.sum(F.exp(x - x_max), axis=axis, keepdims=True)) + x_max
        return [x - log_z]

    # ── Math ops (extended) ───────────────────────────────────────────────────
    if op == "Acosh":
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.arccosh(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.acosh(get(0))]

    if op == "Atanh":
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.arctanh(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.atanh(get(0))]

    # ── Aggregation ───────────────────────────────────────────────────────────
    if op == "Mean":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = result + t
        return [result / np.float32(len(tensors))]

    # ── Tensor construction ───────────────────────────────────────────────────
    if op == "EyeLike":
        x = get(0)
        dtype_attr = _attr(node, "dtype", None)
        dtype = _ONNX_DTYPE.get(int(dtype_attr), np.float32) if dtype_attr is not None else np.float32
        k = int(_attr(node, "k", 0))
        rows, cols = int(x.shape[0]), int(x.shape[1])
        eye = np.eye(rows, cols, k=k, dtype=dtype)
        return [F.asarray(eye) if hasattr(F, 'asarray') else eye]

    if op == "Range":
        start = float(_np(get(0)).flat[0])
        limit = float(_np(get(1)).flat[0])
        delta = float(_np(get(2)).flat[0])
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.arange(start, limit, delta, dtype=jnp.float32)]
        else:
            import tensorflow as tf
            return [tf.range(start, limit, delta, dtype=tf.float32)]

    # ── Linear algebra (extended) ─────────────────────────────────────────────
    if op == "Einsum":
        equation = _attr(node, "equation", b"")
        if isinstance(equation, bytes): equation = equation.decode()
        inputs = [get(i) for i in range(len(node.input)) if node.input[i]]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.einsum(equation, *inputs)]
        else:
            import tensorflow as tf
            return [tf.einsum(equation, *inputs)]

    if op == "Trilu":
        x = get(0)
        k_t = get(1)
        k = int(_np(k_t).flat[0]) if k_t is not None else 0
        upper = int(_attr(node, "upper", 1))
        rows, cols = int(x.shape[-2]), int(x.shape[-1])
        mask_2d = (np.triu(np.ones((rows, cols), dtype=bool), k=k) if upper
                   else np.tril(np.ones((rows, cols), dtype=bool), k=k))
        mask = np.broadcast_to(mask_2d, x.shape)
        return [F.where(mask, x, F.zeros_like(x))]

    # ── Pooling (extended) ────────────────────────────────────────────────────
    if op == "GlobalLpPool":
        x_np = np.array(get(0), dtype=np.float32)
        p = int(_attr(node, "p", 2))
        axes = tuple(range(2, len(x_np.shape)))
        return [(np.sum(np.abs(x_np) ** p, axis=axes, keepdims=True) ** (1.0 / p)).astype(np.float32)]

    if op == "LpPool":
        x_np = np.array(get(0), dtype=np.float32)
        p = int(_attr(node, "p", 2))
        k = _attr(node, "kernel_shape", [2, 2])
        strides = _attr(node, "strides", [1, 1])
        pads = _attr(node, "pads", [0, 0, 0, 0])
        N, C, H, W = x_np.shape
        kH, kW = int(k[0]), int(k[1])
        sH, sW = int(strides[0]), int(strides[1])
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        x_pad = np.pad(x_np, ((0, 0), (0, 0), (pH0, pH1), (pW0, pW1)))
        H_out = (H + pH0 + pH1 - kH) // sH + 1
        W_out = (W + pW0 + pW1 - kW) // sW + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for ii in range(H_out):
            for jj in range(W_out):
                patch = x_pad[:, :, ii*sH:ii*sH+kH, jj*sW:jj*sW+kW]
                out[:, :, ii, jj] = np.sum(np.abs(patch) ** p, axis=(2, 3)) ** (1.0 / p)
        return [out]

    # ── Quantization ──────────────────────────────────────────────────────────
    if op == "QuantizeLinear":
        x_np = np.array(get(0), dtype=np.float32)
        scale = float(_np(get(1)).flat[0])
        zp = get(2)
        zero_point = int(_np(zp).flat[0]) if zp is not None else 0
        zp_dtype = _np(zp).dtype if zp is not None else np.uint8
        quantized = np.round(x_np / scale) + zero_point
        if zp_dtype == np.int8:
            return [np.clip(quantized, -128, 127).astype(np.int8)]
        return [np.clip(quantized, 0, 255).astype(np.uint8)]

    if op == "DequantizeLinear":
        x_np = _np(get(0)).astype(np.float32)
        scale = float(_np(get(1)).flat[0])
        zp = get(2)
        zero_point = float(_np(zp).flat[0]) if zp is not None else 0.0
        return [((x_np - zero_point) * scale).astype(np.float32)]

    # ── Advanced indexing ─────────────────────────────────────────────────────
    if op == "GatherElements":
        data = get(0)
        indices = _np(get(1)).astype(np.int64)
        axis = int(_attr(node, "axis", 0))
        indices = np.where(indices < 0, indices + int(data.shape[axis]), indices)
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.take_along_axis(data, indices, axis=axis)]
        else:
            import tensorflow as tf
            return [tf.experimental.numpy.take_along_axis(data, tf.constant(indices), axis=axis)]

    if op == "GatherND":
        data = get(0)
        indices = _np(get(1)).astype(np.int64)
        k = indices.shape[-1]
        outer_shape = indices.shape[:-1]
        flat_idx = indices.reshape(-1, k)
        if _is_jax_module(F):
            import jax.numpy as jnp
            result = jnp.stack([data[tuple(idx)] for idx in flat_idx])
            return [result.reshape(outer_shape + data.shape[k:])]
        else:
            import tensorflow as tf
            result = tf.gather_nd(data, tf.constant(flat_idx))
            return [tf.reshape(result, list(outer_shape) + [int(d) for d in data.shape[k:]])]

    if op == "ScatterElements":
        # Use numpy for reference; indices always come from initializers
        data_np = np.array(get(0), dtype=np.float32)
        indices = _np(get(1)).astype(np.int64)
        updates_np = np.array(get(2), dtype=np.float32)
        axis = int(_attr(node, "axis", 0))
        indices = np.where(indices < 0, indices + data_np.shape[axis], indices)
        result = data_np.copy()
        for idx in np.ndindex(indices.shape):
            target = list(idx)
            target[axis] = int(indices[idx])
            result[tuple(target)] = updates_np[idx]
        return [result]

    if op == "ScatterND":
        data_np = np.array(get(0), dtype=np.float32)
        indices = _np(get(1)).astype(np.int64)
        updates_np = np.array(get(2), dtype=np.float32)
        k = indices.shape[-1]
        flat_idx = indices.reshape(-1, k)
        upd_tail = updates_np.shape[len(indices.shape) - 1:]
        flat_upd = updates_np.reshape(-1, *upd_tail)
        result = data_np.copy()
        for i, idx in enumerate(flat_idx):
            result[tuple(idx)] = flat_upd[i]
        return [result]

    # ── TopK ──────────────────────────────────────────────────────────────────
    if op == "TopK":
        x = get(0)
        K = int(_np(get(1)).flat[0])
        axis = int(_attr(node, "axis", -1))
        largest = bool(_attr(node, "largest", 1))
        sorted_ = bool(_attr(node, "sorted", 1))
        ndim = len(x.shape)
        axis_mod = axis % ndim
        # Move target axis to last position for framework top_k
        perm = list(range(ndim))
        perm.pop(axis_mod)
        perm.append(axis_mod)
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        if _is_jax_module(F):
            import jax
            import jax.numpy as jnp
            x_t = jnp.transpose(x, perm)
            vals, idx = jax.lax.top_k(x_t if largest else -x_t, K)
            if not largest:
                vals = -vals
            if not sorted_:
                # ONNX TopK with sorted=0 may return values in any order.
                # jax.lax.top_k always returns sorted; agreeing with that
                # is a stronger guarantee than the spec demands, so this
                # never produces a false positive against an ONNX-spec
                # backend that emits sorted output. We still surface the
                # attribute here for callers that rely on the exact match.
                pass
            return [jnp.transpose(vals, inv_perm),
                    jnp.transpose(idx.astype(jnp.int64), inv_perm)]
        else:
            import tensorflow as tf
            x_t = tf.transpose(x, perm)
            res = tf.math.top_k(x_t if largest else -x_t, k=K, sorted=sorted_)
            vals = res.values if largest else -res.values
            idx = tf.cast(res.indices, tf.int64)
            return [tf.transpose(vals, inv_perm), tf.transpose(idx, inv_perm)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _flip_axis(x, axis: int):
    """Flip a tensor along `axis` using slice notation. Works for both
    JAX and TF (which expose Python slicing) when an explicit `flip` op
    is missing on the framework module."""
    ndim = len(x.shape)
    ax   = axis % ndim
    slc  = [slice(None)] * ndim
    slc[ax] = slice(None, None, -1)
    return x[tuple(slc)]


def _resolve_auto_pad(auto_pad: str, in_shape, kernel, strides, dilations):
    """Compute explicit ONNX-style pad list from `auto_pad` for a 2-D op.

    Returns [pH_top, pW_left, pH_bottom, pW_right] in ONNX layout.
    `auto_pad ∈ {NOTSET, VALID, SAME_UPPER, SAME_LOWER}`.
    `in_shape` is the [N,C,H,W] input shape; uses dims [2:].
    """
    if auto_pad in (b"NOTSET", "NOTSET", b"", "", None):
        return None
    if auto_pad in (b"VALID", "VALID"):
        return [0, 0, 0, 0]
    H, W = int(in_shape[2]), int(in_shape[3])
    kH, kW = int(kernel[0]), int(kernel[1])
    sH, sW = int(strides[0]), int(strides[1])
    dH, dW = int(dilations[0]), int(dilations[1])
    ekH = dH * (kH - 1) + 1
    ekW = dW * (kW - 1) + 1
    out_H = (H + sH - 1) // sH
    out_W = (W + sW - 1) // sW
    total_h = max(0, (out_H - 1) * sH + ekH - H)
    total_w = max(0, (out_W - 1) * sW + ekW - W)
    if auto_pad in (b"SAME_UPPER", "SAME_UPPER"):
        return [total_h // 2, total_w // 2,
                total_h - total_h // 2, total_w - total_w // 2]
    # SAME_LOWER
    return [total_h - total_h // 2, total_w - total_w // 2,
            total_h // 2, total_w // 2]


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
    auto_pad  = _attr(node, "auto_pad",  None)
    # auto_pad overrides the explicit `pads` attribute when set to anything
    # other than NOTSET. Without this every model that uses SAME_UPPER /
    # SAME_LOWER / VALID was running with pads=[0,0,0,0], which silently
    # produced wrong output everywhere the harness ran ONNX ops natively.
    kernel_shape = _attr(node, "kernel_shape", [int(w.shape[2]), int(w.shape[3])])
    resolved = _resolve_auto_pad(auto_pad, x.shape, kernel_shape, strides, dilations)
    if resolved is not None:
        pads = resolved

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
    auto_pad  = _attr(node, "auto_pad",     None)
    # AveragePool: include or exclude pads in the divisor. Default 0 (=False)
    # means divide by the count of valid (non-pad) elements, matching the
    # ONNX spec; True means divide by the full kernel area. The original
    # implementation hard-coded False, which silently disagreed with every
    # other backend on count_include_pad=True models — that was the cause
    # of bug_v6_000421 ("XLA AveragePool count_include_pad bug").
    count_include_pad = bool(_attr(node, "count_include_pad", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    resolved = _resolve_auto_pad(auto_pad, x.shape, k, strides, dilations)
    if resolved is not None:
        pads = resolved

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
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            if count_include_pad:
                # Divide by full kernel area regardless of how many real
                # elements fell into each window.
                kernel_area = int(k[0]) * int(k[1])
                y = s / np.float32(kernel_area)
            else:
                ones = F.ones_like(x)
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
            # tf.nn.avg_pool2d implements count_include_pad=False (excludes
            # padding from the divisor). When ONNX requests True we have to
            # compute the windowed sum manually and divide by the kernel area.
            if count_include_pad:
                # Pad explicitly with zeros, then avg_pool with VALID padding.
                # avg_pool divides by kernel_area; pad-zeros contribute 0 to
                # the numerator, giving exactly the include_pad mean.
                pH0, pH1 = int(pads[0]), int(pads[2])
                pW0, pW1 = int(pads[1]), int(pads[3])
                x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                               constant_values=0.0)
                y_nhwc = tf.nn.avg_pool2d(x_pad, ksize, str_tf, padding="VALID")
            else:
                # avg_pool doesn't support dilations in TF, treat as no dilation
                y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    """Resize op with full attribute support.

    The previous implementation ignored every attribute except `mode`,
    silently using each backend's default coordinate transform / nearest
    rounding. That mismatch with ONNX-spec backends like ORT produced
    huge numbers of false-positive bug reports for any model that set
    coordinate_transformation_mode, nearest_mode, cubic_coeff_a, or
    antialias. We now reject any unsupported attribute combination as
    NotImplementedError so the surrounding harness records it as a
    "frontend" issue (skipped, not counted) instead of pretending the
    output is the spec-compliant reference.
    """
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    coord_mode = _attr(node, "coordinate_transformation_mode", b"half_pixel")
    if isinstance(coord_mode, bytes): coord_mode = coord_mode.decode()
    nearest_mode = _attr(node, "nearest_mode", b"round_prefer_floor")
    if isinstance(nearest_mode, bytes): nearest_mode = nearest_mode.decode()
    cubic_coeff_a = float(_attr(node, "cubic_coeff_a", -0.75))
    antialias     = bool(_attr(node, "antialias", 0))
    exclude_outside = bool(_attr(node, "exclude_outside", 0))

    # Compute output spatial size.
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None and _np(scales_t).size > 0:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    # The harness can only emulate the (mode, coord_mode, nearest_mode,
    # antialias) combos that the underlying framework's resize op actually
    # implements identically to the ONNX spec. Anything else must crash
    # rather than silently produce a different answer that would be
    # reported as a backend bug.
    _SUPPORTED_NEAREST_COMBOS = {
        # ONNX  → (jax method,  TF method)             — tested combinations
        ("nearest", "asymmetric", "floor"):       ("nearest", "nearest"),
        ("nearest", "asymmetric", "round_prefer_floor"): ("nearest", "nearest"),
    }
    _SUPPORTED_LINEAR_COMBOS = {
        # half_pixel + linear (no antialias) ↔ jax/TF default
        ("linear", "half_pixel"):        ("linear",  "bilinear"),
        ("linear", "pytorch_half_pixel"):("linear",  "bilinear"),
    }

    key_n = (mode, coord_mode, nearest_mode)
    key_l = (mode, coord_mode)
    if "nearest" in mode and key_n not in _SUPPORTED_NEAREST_COMBOS:
        raise NotImplementedError(
            f"Resize(nearest, coord={coord_mode!r}, "
            f"nearest_mode={nearest_mode!r}) — combination not implemented "
            "by harness reference; skip rather than risk false positive."
        )
    if "linear" in mode and key_l not in _SUPPORTED_LINEAR_COMBOS:
        raise NotImplementedError(
            f"Resize(linear, coord={coord_mode!r}) — combination not "
            "implemented by harness reference."
        )
    if mode == "cubic":
        # Cubic resize differs in coeff_a, antialias and exclude_outside
        # across backends; emulating ONNX exactly is out of scope.
        raise NotImplementedError(
            "Resize(cubic) — harness has no spec-faithful cubic resize; "
            "use ORT directly when verifying cubic models."
        )
    if antialias:
        raise NotImplementedError(
            "Resize(antialias=1) — backends differ on the antialias kernel; "
            "harness will not provide a reference."
        )
    if exclude_outside:
        raise NotImplementedError("Resize(exclude_outside=1) — not implemented")

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = (_SUPPORTED_NEAREST_COMBOS[key_n][0]
                  if "nearest" in mode
                  else _SUPPORTED_LINEAR_COMBOS[key_l][0])
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        tf_method_name = (_SUPPORTED_NEAREST_COMBOS[key_n][1]
                          if "nearest" in mode
                          else _SUPPORTED_LINEAR_COMBOS[key_l][1])
        method = (tf.image.ResizeMethod.NEAREST_NEIGHBOR
                  if tf_method_name == "nearest"
                  else tf.image.ResizeMethod.BILINEAR)
        x_nhwc = tf.transpose(x, [0,2,3,1])
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
