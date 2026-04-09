#!/usr/bin/env python3
"""
Bug #0022 — xla: jax.jit diverges from pytorch_eager.

Patterns  : [['branch', 'three_branch_concat'], ['fusion', 'depthwise_conv_bn_relu'], ['attention', 'matmul_4d_batch'], ['constant', 'redundant_reshape'], ['broadcast', 'tanh_add_mul_chain'], ['layout', 'resize_linear_aligncorners']]
S_diff    : 1.000000   (jit vs pytorch_eager)
Delta_opt : 0.000000   (jit vs no-jit)
Tolerance : 0.001

    python unique_0022.py
Exit 0 = reproduced, 1 = not reproduced.
"""
import base64, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh

TOLERANCE = 0.001

_ONNX_B64 = (
    "CAg62jcKXwoLbW9kZWxfaW5wdXQKCW4wX3RiY193MQoJbjBfdGJjX2IxEgpuMF90YmNfY3YxIgRD"
    "b252KhUKDGtlcm5lbF9zaGFwZUABQAGgAQcqEQoEcGFkc0AAQABAAEAAoAEHCl8KC21vZGVsX2lu"
    "cHV0CgluMF90YmNfdzIKCW4wX3RiY19iMhIKbjBfdGJjX2N2MiIEQ29udioVCgxrZXJuZWxfc2hh"
    "cGVAA0ADoAEHKhEKBHBhZHNAAUABQAFAAaABBwpfCgttb2RlbF9pbnB1dAoJbjBfdGJjX3czCglu"
    "MF90YmNfYjMSCm4wX3RiY19jdjMiBENvbnYqFQoMa2VybmVsX3NoYXBlQAFAAaABByoRCgRwYWRz"
    "QABAAEAAQACgAQcKRQoKbjBfdGJjX2N2MQoKbjBfdGJjX2N2MgoKbjBfdGJjX2N2MxIKbjBfdGJj"
    "X291dCIGQ29uY2F0KgsKBGF4aXMYAaABAgqQAQoKbjBfdGJjX291dAoPbjEwMF9kd2NicmVsdV93"
    "Cg9uMTAwX2R3Y2JyZWx1X2ISEG4xMDBfZHdjYnJlbHVfY3YiBENvbnYqDAoFZ3JvdXAYDKABAioV"
    "CgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUABQAFAAaABByoQCgdzdHJpZGVzQAFAAaAB"
    "BwqnAQoQbjEwMF9kd2NicmVsdV9jdgoWbjEwMF9kd2NicmVsdV9ibl9zY2FsZQoVbjEwMF9kd2Ni"
    "cmVsdV9ibl9iaWFzChVuMTAwX2R3Y2JyZWx1X2JuX21lYW4KFG4xMDBfZHdjYnJlbHVfYm5fdmFy"
    "EhBuMTAwX2R3Y2JyZWx1X2JuIhJCYXRjaE5vcm1hbGl6YXRpb24qEQoHZXBzaWxvbhWsxSc3oAEB"
    "CisKEG4xMDBfZHdjYnJlbHVfYm4SEW4xMDBfZHdjYnJlbHVfb3V0IgRSZWx1CjcKEW4xMDBfZHdj"
    "YnJlbHVfb3V0CgtuMjAwX21tNGRfdxINbjIwMF9tbTRkX291dCIGTWF0TXVsCjUKDW4yMDBfbW00"
    "ZF9vdXQKDW4zMDBfcnJzaF9zZmwSDG4zMDBfcnJzaF9yMSIHUmVzaGFwZQo1CgxuMzAwX3Jyc2hf"
    "cjEKDW4zMDBfcnJzaF9zb3ISDW4zMDBfcnJzaF9vdXQiB1Jlc2hhcGUKIgoNbjMwMF9ycnNoX291"
    "dBILbjQwMF90YW1fdGgiBFRhbmgKKwoLbjQwMF90YW1fdGgKCm40MDBfdGFtX2MSC240MDBfdGFt"
    "X2FkIgNBZGQKLwoLbjQwMF90YW1fYWQKDW4zMDBfcnJzaF9vdXQSDG40MDBfdGFtX291dCIDTXVs"
    "Co0BCgxuNDAwX3RhbV9vdXQKDW41MDBfcmVzaV9yb2kKEG41MDBfcmVzaV9zY2FsZXMSDW41MDBf"
    "cmVzaV9vdXQiBlJlc2l6ZSoyCh5jb29yZGluYXRlX3RyYW5zZm9ybWF0aW9uX21vZGUiDWFsaWdu"
    "X2Nvcm5lcnOgAQMqEQoEbW9kZSIGbGluZWFyoAEDEgt0cmlvbl9ncmFwaCpHCAQIAwgBCAEQAUIJ"
    "bjBfdGJjX3cxSjCrQTq+nAUSPQTygz/We/c+/A03vxynR7/1rwY/x96aO1A/uL8yhUo/SbQvP2rm"
    "iz4qIQgEEAFCCW4wX3RiY19iMUoQAAAAAAAAAAAAAAAAAAAAACrIAwgECAMIAwgDEAFCCW4wX3Ri"
    "Y193MkqwA/ab+D5cAqe+hKdAPl4OCj45Wao+JH7EPgwfVT4V5Eu9v37lPb2wBL3pPl6+3weFPtEr"
    "iL7ouke+RCGiPfN4Ub5euFa+nyKuvtg+CD47CGg+Qn/uPrsmC7zT+6C+/FAoPqtGqD4aqVy+YEdD"
    "vq/T2r7SgBG+9QU4Pjh9Ab6fl++9tJ6Rvp8e4j7C4OU9E7vRPqndDb+H7p49f5WGvA9qFr3Ixf29"
    "kKDCvrwGaj26vJe+4beBvh7bgztI8nS9WPugPaKYnb0VkXM+Wh/ivaFO/b4ONx0+aYRHPqcR7r7e"
    "jh6+5XCpvtOqLj2HJTQ+d5OwvYBhgr4SGcG9WHsovuLMvL34dNA9MQTNPR4c7765PwG/yJR+Pqan"
    "Xr4a3go9AF81vvyGV76Nr/e+yRWrPqfukz4PALU+1XkHP6NZnT3sgae+lR3MvheUsD4FDdg92Ulg"
    "vhKnf76U7oy++i24Pr7IzD7b7Om+LksOPpnChL6gax8+2i9AvvjrzL6i+WU+G3Kwvf9Rm74Krhs+"
    "dly0vQX8oj3TBgI+YXAwP318UT76aAy+biGGvtKN6r2gVa6+qnxqviohCAQQAUIJbjBfdGJjX2Iy"
    "ShAAAAAAAAAAAAAAAAAAAAAAKkcIBAgDCAEIARABQgluMF90YmNfdzNKMKtBOr6cBRI9BPKDP9Z7"
    "9z78DTe/HKdHv/WvBj/H3po7UD+4vzKFSj9JtC8/auaLPiohCAQQAUIJbjBfdGJjX2IzShAAAAAA"
    "AAAAAAAAAAAAAAAAKs4DCAwIAQgDCAMQAUIPbjEwMF9kd2NicmVsdV93SrAD71vxPu9b8T7vW/E+"
    "71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vxvu9b8b7vW/G+71vxvu9b8b7vW/G+71vxvu9b8b7v"
    "W/G+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vxvu9b8b7vW/G+71vxvu9b"
    "8b7vW/G+71vxvu9b8b7vW/G+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vx"
    "vu9b8b7vW/G+71vxvu9b8b7vW/G+71vxvu9b8b7vW/G+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+"
    "71vxPu9b8T7vW/E+71vxvu9b8b7vW/G+71vxvu9b8b7vW/G+71vxvu9b8b7vW/G+71vxPu9b8T7v"
    "W/E+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vxvu9b8b7vW/G+71vxvu9b8b7vW/G+71vxvu9b"
    "8b7vW/G+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vxPu9b8T7vW/E+71vxvu9b8b7vW/G+71vx"
    "vu9b8b7vW/G+71vxvu9b8b7vW/G+KkcIDBABQg9uMTAwX2R3Y2JyZWx1X2JKMAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACpOCAwQAUIWbjEwMF9kd2NicmVs"
    "dV9ibl9zY2FsZUowAACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8A"
    "AIA/Kk0IDBABQhVuMTAwX2R3Y2JyZWx1X2JuX2JpYXNKMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "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"
    "CAwKAghACgIIQEIECgAQEQ=="
)
_INPUTS = {
    'model_input': (
    "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",
        [1, 3, 32, 32],
    ),
}

# ── Inlined ONNX op dispatcher (no project imports needed) ────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────


def _decode():
    model_bytes = base64.b64decode(_ONNX_B64)
    inputs = {n: np.frombuffer(base64.b64decode(b64), dtype=np.float32
                               ).copy().reshape(sh)
              for n, (b64, sh) in _INPUTS.items()}
    return model_bytes, inputs

def _rel_l2(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape: return float("inf")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))

def _pytorch_eager(model_bytes, inputs):
    """Reference: onnx2torch eager (no compiler) — same as oracle pytorch_eager."""
    import torch, onnx2torch
    model = onnx.load_from_string(model_bytes)
    m = onnx2torch.convert(model).eval()
    inp = next(iter(inputs.values()))
    x = torch.from_numpy(inp)
    with torch.no_grad():
        out = m(x)
    if isinstance(out, (list, tuple)): out = out[0]
    return out.detach().cpu().float().numpy().ravel()


def _jax_run(model, x_np, use_jit):
    import jax, jax.numpy as jnp
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    init_np  = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x_jax):
        vals = dict(init_np)
        vals[inp_name] = x_jax
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    if use_jit:
        return np.array(jax.jit(fn)(jnp.array(x_np)), dtype=np.float32).ravel()
    with jax.disable_jit():
        return np.array(fn(jnp.array(x_np)), dtype=np.float32).ravel()


def main():
    model_bytes, inputs = _decode()
    model  = onnx.load_from_string(model_bytes)
    x_np   = next(iter(inputs.values()))

    # Reference: pytorch eager
    try:
        ref = _pytorch_eager(model_bytes, inputs)
    except Exception as exc:
        print(f"[warn] pytorch_eager reference failed: {exc}")
        ref = None

    # Target: jax.jit
    try:
        target    = _jax_run(model, x_np, use_jit=True)
        noopt_out = _jax_run(model, x_np, use_jit=False)
    except Exception as exc:
        print(f"[crash] xla: {exc}")
        sys.exit(0)

    diff_vs_ref    = _rel_l2(target, ref) if ref is not None else 0.0
    diff_opt_noopt = _rel_l2(target, noopt_out)

    print(f"rel L2(jit vs pytorch_eager) = {diff_vs_ref:.6e}")
    print(f"rel L2(jit vs no-jit)        = {diff_opt_noopt:.6e}")
    print(f"tolerance = {TOLERANCE}")

    if diff_vs_ref > TOLERANCE or diff_opt_noopt > TOLERANCE:
        print("BUG REPRODUCED")
        sys.exit(0)
    print("not reproduced")
    sys.exit(1)

if __name__ == "__main__":
    main()
