#!/usr/bin/env python3
"""Emit clean minimal repros: one self-contained .py per bug.

ORT bugs   : load the real .onnx file, ort_noopt vs ort_opt. (~40 lines)
JAX bugs   : generate NATIVE JAX code — no ONNX loading, no dispatch_op.
             Weights embedded as hex numpy arrays; ops emitted as jax/lax calls.
TC bugs    : load the real .onnx file, onnx2torch eager vs torch.compile.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parent.parent

_ONNX_DTYPE = {
    TensorProto.FLOAT:  "np.float32",
    TensorProto.DOUBLE: "np.float64",
    TensorProto.INT32:  "np.int32",
    TensorProto.INT64:  "np.int64",
    TensorProto.BOOL:   "np.bool_",
    TensorProto.UINT8:  "np.uint8",
    TensorProto.INT8:   "np.int8",
}


# ── Attribute helpers ─────────────────────────────────────────────────────────

def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(a.t)
    return default


# ── Variable naming ───────────────────────────────────────────────────────────

def _vname(onnx_name: str) -> str:
    """Convert an ONNX tensor name to a safe Python variable name."""
    return re.sub(r'[^a-zA-Z0-9]', '_', onnx_name).strip('_') or '_v'


# ── JAX code generator ────────────────────────────────────────────────────────

class _NotSupported(Exception):
    pass


def _gen_op(op: str, node, ins: list[Optional[str]],
            shapes: dict[str, tuple],
            init_shapes: dict[str, tuple]) -> str:
    """
    Return a Python expression (JAX) for this op.
    ins[i] = Python variable name for input i (or None if optional/absent).
    Raises _NotSupported if this op can't be generated.
    """
    def i(k): return ins[k] if k < len(ins) else None

    # ── Element-wise arithmetic ───────────────────────────────────────────────
    if op == "Add":       return f"{i(0)} + {i(1)}"
    if op == "Sub":       return f"{i(0)} - {i(1)}"
    if op == "Mul":       return f"{i(0)} * {i(1)}"
    if op == "Div":       return f"{i(0)} / {i(1)}"
    if op == "Neg":       return f"-{i(0)}"
    if op == "Abs":       return f"jnp.abs({i(0)})"
    if op == "Sqrt":      return f"jnp.sqrt({i(0)})"
    if op == "Exp":       return f"jnp.exp({i(0)})"
    if op == "Log":       return f"jnp.log({i(0)})"
    if op == "Tanh":      return f"jnp.tanh({i(0)})"
    if op == "Reciprocal":return f"np.float32(1.0) / {i(0)}"
    if op == "Sin":       return f"jnp.sin({i(0)})"
    if op == "Cos":       return f"jnp.cos({i(0)})"
    if op == "Floor":     return f"jnp.floor({i(0)})"
    if op == "Ceil":      return f"jnp.ceil({i(0)})"
    if op == "Round":     return f"jnp.round({i(0)})"
    if op == "Sign":      return f"jnp.sign({i(0)})"
    if op == "Softplus":  return f"jnp.log(np.float32(1.0) + jnp.exp({i(0)}))"
    if op == "Sigmoid":   return f"(np.float32(1.0) / (np.float32(1.0) + jnp.exp(-{i(0)})))"

    if op == "Pow":
        exp = i(1)
        # exponent is often a scalar initializer
        return f"{i(0)} ** np.float32({i(1)}.flat[0])" if exp else f"{i(0)} ** {i(1)}"

    if op == "Erf":
        return f"jax.scipy.special.erf({i(0)})"

    if op in ("Max", "Min"):
        fn = "jnp.maximum" if op == "Max" else "jnp.minimum"
        result = i(0)
        for k in range(1, len(ins)):
            if ins[k]:
                result = f"{fn}({result}, {ins[k]})"
        return result

    # ── Activations ───────────────────────────────────────────────────────────
    if op == "Relu":
        return f"jnp.maximum({i(0)}, np.float32(0.0))"

    if op == "LeakyRelu":
        alpha = float(_attr(node, "alpha", 0.01))
        return f"jnp.where({i(0)} >= np.float32(0.0), {i(0)}, np.float32({alpha!r}) * {i(0)})"

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        return (f"jnp.where({i(0)} >= np.float32(0.0), {i(0)}, "
                f"np.float32({alpha!r}) * (jnp.exp({i(0)}) - np.float32(1.0)))")

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        return (f"np.float32({gamma!r}) * jnp.where({i(0)} >= np.float32(0.0), {i(0)}, "
                f"np.float32({alpha!r}) * (jnp.exp({i(0)}) - np.float32(1.0)))")

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta", 0.5))
        return (f"jnp.clip(np.float32({alpha!r}) * {i(0)} + np.float32({beta!r}), "
                f"np.float32(0.0), np.float32(1.0))")

    if op == "HardSwish":
        return (f"{i(0)} * jnp.clip({i(0)} / np.float32(6.0) + np.float32(0.5), "
                f"np.float32(0.0), np.float32(1.0))")

    if op == "Mish":
        return f"{i(0)} * jnp.tanh(jnp.log(np.float32(1.0) + jnp.exp({i(0)})))"

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = i(0)
        return (f"(lambda _x: _x / jnp.sum(_x, axis={axis}, keepdims=True))"
                f"(jnp.exp({x} - jnp.max({x}, axis={axis}, keepdims=True)))")

    if op == "Clip":
        x = i(0); mn = i(1); mx = i(2)
        parts = [x]
        if mn:
            parts = [f"jnp.maximum({parts[0]}, jnp.asarray({mn}, dtype={parts[0]}.dtype))"]
        if mx:
            parts = [f"jnp.minimum({parts[0]}, jnp.asarray({mx}, dtype={parts[0]}.dtype))"]
        return parts[0]

    if op in ("Identity", "Dropout"):
        return i(0)

    if op == "Cast":
        to    = int(_attr(node, "to", TensorProto.FLOAT))
        dtype = _ONNX_DTYPE.get(to, "np.float32")
        return f"{i(0)}.astype({dtype})"

    # ── Shape ops ─────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        if perm is None:
            return f"jnp.transpose({i(0)})"
        return f"jnp.transpose({i(0)}, {list(perm)})"

    if op == "Reshape":
        # shape tensor must be int — cast to be safe
        sh_var = i(1)
        return f"jnp.reshape({i(0)}, [int(v) for v in {sh_var}.flat])"

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = i(0)
        return (f"(lambda _x: jnp.reshape(_x, [int(np.prod(_x.shape[:{axis}])), "
                f"int(np.prod(_x.shape[{axis}:]))]))"
                f"({x})")

    if op == "Unsqueeze":
        axes = _attr(node, "axes", None)
        if axes is None:
            axes_val = i(1)
            return f"(lambda _x, _ax: jnp.expand_dims(_x, axis=int(_ax.flat[0])))({i(0)}, {axes_val})"
        x = i(0)
        for ax in sorted(int(a) for a in axes):
            x = f"jnp.expand_dims({x}, axis={ax})"
        return x

    if op == "Squeeze":
        axes = _attr(node, "axes", None)
        x = i(0)
        if axes:
            for ax in sorted((int(a) for a in axes), reverse=True):
                x = f"jnp.squeeze({x}, axis={ax})"
            return x
        return f"jnp.squeeze({x})"

    if op == "Expand":
        sh = i(1)
        return f"jnp.broadcast_to({i(0)}, {sh}.tolist())"

    if op == "Gather":
        axis = int(_attr(node, "axis", 0))
        return f"jnp.take({i(0)}, {i(1)}.astype(np.int32), axis={axis})"

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [v for v in ins if v]
        return f"jnp.concatenate([{', '.join(tensors)}], axis={axis})"

    if op == "Slice":
        x = i(0)
        starts = i(1); ends = i(2); axes_v = i(3); steps_v = i(4)
        # All slice inputs are static initializers; build slice at codegen time
        # We can't statically evaluate these easily — emit runtime slicing
        return (f"(lambda _x, _s, _e, _ax, _st: _x[tuple("
                f"slice(int(_s[k]),int(_e[k]) if abs(int(_e[k]))<2**30 else None,int(_st[k])) "
                f"if k < len(_ax) else slice(None) "
                f"for k in range(len(_x.shape)))])"
                f"({x}, {starts}, {ends}, "
                f"{axes_v if axes_v else f'np.arange(len({x}.shape))'}, "
                f"{steps_v if steps_v else 'np.ones(len('+starts+'), dtype=np.int64)'})")

    if op == "Pad":
        x = i(0); pads_v = i(1)
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        return (f"jnp.pad({x}, "
                f"[(int({pads_v}[k]), int({pads_v}[k+len({x}.shape)])) "
                f"for k in range(len({x}.shape))], "
                f"mode={mode!r})")

    if op == "Tile":
        return f"jnp.tile({i(0)}, {i(1)}.tolist())"

    # ── Linear algebra ────────────────────────────────────────────────────────
    if op == "MatMul":
        return f"jnp.matmul({i(0)}, {i(1)})"

    if op == "Gemm":
        A = i(0); B = i(1); C = i(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta", 1.0))
        tA = _attr(node, "transA", 0)
        tB = _attr(node, "transB", 0)
        if tA: A = f"jnp.swapaxes({A}, -1, -2)"
        if tB: B = f"jnp.swapaxes({B}, -1, -2)"
        expr = f"np.float32({alpha!r}) * jnp.matmul({A}, {B})"
        if C:
            expr += f" + np.float32({beta!r}) * {C}"
        return expr

    # ── Convolution ───────────────────────────────────────────────────────────
    if op == "Conv":
        x = i(0); w = i(1); b = i(2)
        pads      = list(_attr(node, "pads",      [0,0,0,0]))
        strides   = list(_attr(node, "strides",   [1,1]))
        dilations = list(_attr(node, "dilations", [1,1]))
        group     = int(_attr(node, "group", 1))
        padding   = ((int(pads[0]),int(pads[2])), (int(pads[1]),int(pads[3])))
        expr = (f"lax.conv_general_dilated({x}, {w}, "
                f"window_strides={[int(s) for s in strides]}, "
                f"padding={padding}, "
                f"lhs_dilation=(1,1), "
                f"rhs_dilation={[int(d) for d in dilations]}, "
                f"dimension_numbers=('NCHW','OIHW','NCHW'), "
                f"feature_group_count={group})")
        if b:
            expr = f"({expr} + jnp.reshape({b}, [1,-1,1,1]))"
        return expr

    if op == "ConvTranspose":
        raise _NotSupported("ConvTranspose")  # complex; fall back to dispatch

    # ── Normalization ─────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = i(0); scale = i(1); B_ = i(2); mean = i(3); var = i(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        return (f"(lambda _x, _s, _b, _m, _v: "
                f"jnp.reshape(_s, [1,-1]+[1]*(_x.ndim-2)) * "
                f"((_x - jnp.reshape(_m, [1,-1]+[1]*(_x.ndim-2))) / "
                f"jnp.sqrt(jnp.reshape(_v, [1,-1]+[1]*(_x.ndim-2)) + np.float32({eps!r}))) + "
                f"jnp.reshape(_b, [1,-1]+[1]*(_x.ndim-2)))"
                f"({x}, {scale}, {B_}, {mean}, {var})")

    if op == "InstanceNormalization":
        x = i(0); scale = i(1); B_ = i(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        return (f"(lambda _x, _s, _b: "
                f"(lambda _m, _v: "
                f"jnp.reshape(_s, [1,-1]+[1]*(_x.ndim-2)) * "
                f"((_x - _m) / jnp.sqrt(_v + np.float32({eps!r}))) + "
                f"jnp.reshape(_b, [1,-1]+[1]*(_x.ndim-2)))"
                f"(jnp.mean(_x, axis=tuple(range(2,_x.ndim)), keepdims=True), "
                f"jnp.mean((_x - jnp.mean(_x, axis=tuple(range(2,_x.ndim)), keepdims=True))**2, "
                f"axis=tuple(range(2,_x.ndim)), keepdims=True)))"
                f"({x}, {scale}, {B_})")

    if op == "LayerNormalization":
        x = i(0); scale = i(1); B_ = i(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        expr = (f"(lambda _x: "
                f"(lambda _m, _v: (_x - _m) / jnp.sqrt(_v + np.float32({eps!r})))"
                f"(jnp.mean(_x, axis=tuple(range({axis}%_x.ndim, _x.ndim)), keepdims=True), "
                f"jnp.mean((_x - jnp.mean(_x, axis=tuple(range({axis}%_x.ndim, _x.ndim)), keepdims=True))**2, "
                f"axis=tuple(range({axis}%_x.ndim, _x.ndim)), keepdims=True)))"
                f"({x})")
        if scale:
            expr = f"({expr}) * {scale}"
        if B_:
            expr = f"({expr}) + {B_}"
        return expr

    # ── Pooling ───────────────────────────────────────────────────────────────
    if op == "GlobalAveragePool":
        return f"jnp.mean({i(0)}, axis=tuple(range(2, {i(0)}.ndim)), keepdims=True)"

    if op == "GlobalMaxPool":
        return f"jnp.max({i(0)}, axis=tuple(range(2, {i(0)}.ndim)), keepdims=True)"

    if op in ("MaxPool", "AveragePool"):
        x = i(0)
        k         = list(_attr(node, "kernel_shape", [2,2]))
        strides   = list(_attr(node, "strides", [1,1]))
        pads      = list(_attr(node, "pads", [0,0,0,0]))
        dilations = list(_attr(node, "dilations", [1,1]))
        ceil_mode = int(_attr(node, "ceil_mode", 0))
        pH0, pH1  = int(pads[0]), int(pads[2])
        pW0, pW1  = int(pads[1]), int(pads[3])
        padding   = f"((0,0),(0,0),({pH0},{pH1}),({pW0},{pW1}))"
        window    = f"(1, 1, {int(k[0])}, {int(k[1])})"
        strides_t = f"(1, 1, {int(strides[0])}, {int(strides[1])})"
        win_dil   = f"(1, 1, {int(dilations[0])}, {int(dilations[1])})"
        if op == "MaxPool":
            return (f"lax.reduce_window({x}, -jnp.inf, lax.max, "
                    f"{window}, {strides_t}, {padding}, window_dilation={win_dil})")
        else:
            # AveragePool: sum / count
            return (f"(lambda _x: "
                    f"lax.reduce_window(_x, 0.0, lax.add, {window}, {strides_t}, {padding}, window_dilation={win_dil}) / "
                    f"lax.reduce_window(jnp.ones_like(_x), 0.0, lax.add, {window}, {strides_t}, {padding}, window_dilation={win_dil}))"
                    f"({x})")

    # ── Reductions ────────────────────────────────────────────────────────────
    if op in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceL2"):
        x = i(0)
        axes = _attr(node, "axes", None)
        if axes is None and i(1):
            axes = f"tuple({i(1)}.tolist())"
        else:
            axes = tuple(int(a) for a in axes) if axes else None
        kd = bool(_attr(node, "keepdims", 1))
        ax_repr = repr(axes)
        if op == "ReduceMean": return f"jnp.mean({x}, axis={ax_repr}, keepdims={kd})"
        if op == "ReduceSum":  return f"jnp.sum({x}, axis={ax_repr}, keepdims={kd})"
        if op == "ReduceMax":  return f"jnp.max({x}, axis={ax_repr}, keepdims={kd})"
        if op == "ReduceL2":
            return f"jnp.sqrt(jnp.sum({x}*{x}, axis={ax_repr}, keepdims={kd}))"

    # ── Misc ──────────────────────────────────────────────────────────────────
    if op == "Where":
        return f"jnp.where({i(0)}, {i(1)}, {i(2)})"

    if op == "Resize":
        x = i(0); scales_v = i(2); sizes_v = i(3)
        mode = _attr(node, "mode", b"nearest")
        if isinstance(mode, bytes): mode = mode.decode()
        method = "nearest" if "nearest" in mode else "linear"
        if sizes_v:
            return (f"(lambda _x, _sz: jnp.transpose("
                    f"jax.image.resize(jnp.transpose(_x,(0,2,3,1)), "
                    f"(_x.shape[0], int(_sz[2]), int(_sz[3]), _x.shape[1]), method={method!r}), "
                    f"(0,3,1,2)))"
                    f"({x}, {sizes_v})")
        else:
            return (f"(lambda _x, _sc: jnp.transpose("
                    f"jax.image.resize(jnp.transpose(_x,(0,2,3,1)), "
                    f"(int(_x.shape[0]), int(_x.shape[2]*_sc[2]), int(_x.shape[3]*_sc[3]), int(_x.shape[1])), "
                    f"method={method!r}), (0,3,1,2)))"
                    f"({x}, {scales_v})")

    if op == "ConstantOfShape":
        sh = i(0)
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return f"jnp.full({sh}.tolist(), np.float32({val!r}))"

    if op == "Shape":
        return f"np.array({i(0)}.shape, dtype=np.int64)"

    if op in ("Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "Not"):
        a, b = i(0), i(1)
        if op == "Equal":          return f"({a} == {b})"
        if op == "Less":           return f"({a} < {b})"
        if op == "Greater":        return f"({a} > {b})"
        if op == "LessOrEqual":    return f"({a} <= {b})"
        if op == "GreaterOrEqual": return f"({a} >= {b})"
        if op == "Not":            return f"~{a}"

    if op == "CumSum":
        axis = f"int({i(1)}.flat[0])" if i(1) else "0"
        return f"jnp.cumsum({i(0)}, axis={axis})"

    if op == "DepthToSpace":
        x = i(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        if mode == "DCR":
            return (f"(lambda _x: jnp.reshape(jnp.transpose("
                    f"jnp.reshape(_x, [_x.shape[0], {bs}, {bs}, _x.shape[1]//({bs}*{bs}), _x.shape[2], _x.shape[3]]), "
                    f"[0,3,4,1,5,2]), "
                    f"[_x.shape[0], _x.shape[1]//({bs*bs}), _x.shape[2]*{bs}, _x.shape[3]*{bs}]))"
                    f"({x})")
        else:
            return (f"(lambda _x: jnp.reshape(jnp.transpose("
                    f"jnp.reshape(_x, [_x.shape[0], _x.shape[1]//({bs*bs}), {bs}, {bs}, _x.shape[2], _x.shape[3]]), "
                    f"[0,1,4,2,5,3]), "
                    f"[_x.shape[0], _x.shape[1]//({bs*bs}), _x.shape[2]*{bs}, _x.shape[3]*{bs}]))"
                    f"({x})")

    # ── Activations ──────────────────────────────────────────────────────────────
    if op == "LeakyRelu":
        alpha = float(_attr(node, "alpha", 0.01))
        return f"jnp.where({i(0)} >= 0, {i(0)}, np.float32({alpha!r}) * {i(0)})"
    if op == "Celu":
        alpha = float(_attr(node, "alpha", 1.0))
        return f"jnp.maximum(0.0, {i(0)}) + jnp.minimum(0.0, np.float32({alpha!r}) * (jnp.exp({i(0)} / np.float32({alpha!r})) - 1.0))"
    if op == "Softsign":
        return f"({i(0)} / (np.float32(1.0) + jnp.abs({i(0)})))"
    if op == "Dropout":
        return i(0)  # inference mode: identity
    if op == "Asin":
        return f"jnp.arcsin({i(0)})"
    if op == "Acos":
        return f"jnp.arccos({i(0)})"
    if op == "Atan":
        return f"jnp.arctan({i(0)})"
    if op == "Erf":
        return f"jax.scipy.special.erf({i(0)})"

    # ── Shape manipulation ────────────────────────────────────────────────────────
    if op == "Squeeze":
        x = i(0)
        axes_inp = i(1)
        if axes_inp:
            return f"jnp.squeeze({x}, axis=tuple(int(a) for a in {axes_inp}.flat))"
        axes_attr = _attr(node, "axes", None)
        if axes_attr is not None:
            return f"jnp.squeeze({x}, axis={tuple(int(a) for a in axes_attr)})"
        return f"jnp.squeeze({x})"
    if op == "Unsqueeze":
        x = i(0)
        axes_inp = i(1)
        if axes_inp:
            # apply unsqueeze one axis at a time in sorted order
            return (f"(lambda _x, _axes: "
                    f"__import__('functools').reduce(lambda a, ax: jnp.expand_dims(a, axis=int(ax)), sorted(_axes.flat), _x))"
                    f"({x}, {axes_inp})")
        axes_attr = list(int(a) for a in _attr(node, "axes", []))
        expr = x
        for ax in sorted(axes_attr):
            expr = f"jnp.expand_dims({expr}, axis={ax})"
        return expr
    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        return f"jnp.reshape({i(0)}, [{i(0)}.shape[0], -1])" if axis == 1 else f"jnp.reshape({i(0)}, [-1])"
    if op == "Expand":
        return f"jnp.broadcast_to({i(0)}, {i(1)}.tolist())"
    if op == "Tile":
        return f"jnp.tile({i(0)}, {i(1)}.tolist())"

    # ── Indexing ──────────────────────────────────────────────────────────────────
    if op == "Gather":
        axis = int(_attr(node, "axis", 0))
        return f"jnp.take({i(0)}, {i(1)}, axis={axis})"
    if op == "GatherElements":
        axis = int(_attr(node, "axis", 0))
        return f"jnp.take_along_axis({i(0)}, {i(1)}, axis={axis})"
    if op == "Slice":
        x = i(0); starts = i(1); ends = i(2); axes = i(3); steps = i(4)
        return (f"(lambda _x, _st, _en, _ax, _sp: "
                f"_x[tuple(slice(int(_st[k]), int(_en[k]), int(_sp[k]) if _sp is not None else 1) "
                f"if k in [int(a) for a in _ax.flat] else slice(None) "
                f"for k in range(_x.ndim))])"
                f"({x}, {starts}, {ends}, {axes if axes else 'np.arange(' + x + '.ndim)'}, {steps if steps else 'None'})")

    # ── Linear algebra ────────────────────────────────────────────────────────────
    if op == "Gemm":
        A = i(0); B = i(1); C = i(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        transA = int(_attr(node, "transA", 0))
        transB = int(_attr(node, "transB", 0))
        a_expr = f"{A}.T" if transA else A
        b_expr = f"{B}.T" if transB else B
        expr = f"np.float32({alpha!r}) * jnp.matmul({a_expr}, {b_expr})"
        if C:
            expr += f" + np.float32({beta!r}) * {C}"
        return expr
    if op == "Einsum":
        eq = _attr(node, "equation", b"").decode() if isinstance(_attr(node, "equation", b""), bytes) else _attr(node, "equation", "")
        args = ", ".join(ins[k] for k in range(len(ins)) if ins[k])
        return f"jnp.einsum({eq!r}, {args})"

    # ── Type / value ops ─────────────────────────────────────────────────────────
    if op == "Cast":
        to = int(_attr(node, "to", TensorProto.FLOAT))
        dtype_map = {TensorProto.FLOAT: "np.float32", TensorProto.DOUBLE: "np.float64",
                     TensorProto.INT32: "np.int32", TensorProto.INT64: "np.int64",
                     TensorProto.BOOL: "np.bool_"}
        dtype = dtype_map.get(to, "np.float32")
        return f"{i(0)}.astype({dtype})"
    if op == "Clip":
        mn = i(1); mx = i(2)
        x = i(0)
        if mn and mx:
            return f"jnp.clip({x}, {mn}.item() if hasattr({mn}, 'item') else float({mn}), {mx}.item() if hasattr({mx}, 'item') else float({mx}))"
        if mn:
            return f"jnp.maximum({x}, {mn}.item() if hasattr({mn}, 'item') else float({mn}))"
        if mx:
            return f"jnp.minimum({x}, {mx}.item() if hasattr({mx}, 'item') else float({mx}))"
        return x
    if op == "Pad":
        x = i(0); pads_v = i(1)
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        if mode == "constant":
            return (f"(lambda _x, _p: jnp.pad(_x, "
                    f"[(int(_p[k]), int(_p[k+_x.ndim])) for k in range(_x.ndim)]))"
                    f"({x}, {pads_v})")
        return f"jnp.pad({x}, [(0,0)]*{x}.ndim)"  # safe fallback
    if op == "Mod":
        fmod = int(_attr(node, "fmod", 0))
        if fmod:
            return f"jnp.fmod({i(0)}, {i(1)})"
        return f"jnp.mod({i(0)}, {i(1)})"
    if op == "Range":
        return f"jnp.arange(float({i(0)}.flat[0]), float({i(1)}.flat[0]), float({i(2)}.flat[0]), dtype=np.float32)"

    # ── Reductions (extra) ────────────────────────────────────────────────────────
    if op in ("ReduceL1", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare", "ReduceProd"):
        x = i(0)
        axes = _attr(node, "axes", None)
        if axes is None and i(1):
            axes = f"tuple({i(1)}.tolist())"
        else:
            axes = tuple(int(a) for a in axes) if axes else None
        kd = bool(_attr(node, "keepdims", 1))
        ax_repr = repr(axes)
        if op == "ReduceL1":        return f"jnp.sum(jnp.abs({x}), axis={ax_repr}, keepdims={kd})"
        if op == "ReduceLogSum":    return f"jnp.log(jnp.sum({x}, axis={ax_repr}, keepdims={kd}))"
        if op == "ReduceLogSumExp": return f"jax.scipy.special.logsumexp({x}, axis={ax_repr}, keepdims={kd})"
        if op == "ReduceSumSquare": return f"jnp.sum({x}*{x}, axis={ax_repr}, keepdims={kd})"
        if op == "ReduceProd":      return f"jnp.prod({x}, axis={ax_repr}, keepdims={kd})"

    # ── Misc ──────────────────────────────────────────────────────────────────────
    if op == "TopK":
        x = i(0); K = i(1)
        axis = int(_attr(node, "axis", -1))
        largest = int(_attr(node, "largest", 1))
        return (f"(lambda _x, _k: (jnp.take_along_axis(_x, "
                f"jnp.argsort(_x, axis={axis})[..., ::-1][..., :int(_k.flat[0])] if {largest} "
                f"else jnp.argsort(_x, axis={axis})[..., :int(_k.flat[0])], axis={axis}), "
                f"jnp.argsort(_x, axis={axis})[..., ::-1][..., :int(_k.flat[0])] if {largest} "
                f"else jnp.argsort(_x, axis={axis})[..., :int(_k.flat[0])]))"
                f"({x}, {K})")
    if op == "Sum":
        return " + ".join(ins[k] for k in range(len(ins)) if ins[k])
    if op == "Max":
        args = ", ".join(ins[k] for k in range(len(ins)) if ins[k])
        if len([k for k in range(len(ins)) if ins[k]]) == 1:
            return i(0)
        return f"jnp.maximum({', '.join(ins[k] for k in range(len(ins)) if ins[k])})"
    if op == "Min":
        args = [ins[k] for k in range(len(ins)) if ins[k]]
        if len(args) == 1: return args[0]
        return f"jnp.minimum({args[0]}, {args[1]})"
    if op == "Mean":
        args = [ins[k] for k in range(len(ins)) if ins[k]]
        return f"({' + '.join(args)}) / np.float32({len(args)})"
    if op == "ReduceMin":
        x = i(0)
        axes = _attr(node, "axes", None)
        if axes is None and i(1):
            axes = f"tuple({i(1)}.tolist())"
        else:
            axes = tuple(int(a) for a in axes) if axes else None
        kd = bool(_attr(node, "keepdims", 1))
        return f"jnp.min({x}, axis={repr(axes)}, keepdims={kd})"
    if op == "PRelu":
        return f"jnp.where({i(0)} >= 0, {i(0)}, {i(1)} * {i(0)})"
    if op == "HardSwish":
        return f"(lambda _x: _x * jnp.clip(_x / 6.0 + 0.5, 0.0, 1.0))({i(0)})"
    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        return f"jnp.clip({i(0)} * np.float32({alpha!r}) + np.float32({beta!r}), 0.0, 1.0)"
    if op == "Mish":
        return f"(lambda _x: _x * jnp.tanh(jnp.log(np.float32(1.0) + jnp.exp(_x))))({i(0)})"
    if op == "LRN":
        x = i(0)
        size  = int(_attr(node, "size", 5))
        alpha = float(_attr(node, "alpha", 1e-4))
        beta  = float(_attr(node, "beta",  0.75))
        bias  = float(_attr(node, "bias",  1.0))
        # LRN: across-channel normalization
        return (f"(lambda _x: _x / jnp.power("
                f"np.float32({bias!r}) + np.float32({alpha!r}/{size}) * "
                f"lax.reduce_window(_x**2, 0.0, lax.add, "
                f"(1,{size},1,1), (1,1,1,1), 'SAME'), np.float32({beta!r})))({x})")
    if op == "Identity":
        return i(0)

    if op in ("Split",):
        raise _NotSupported(op)  # multi-output; handled separately below

    raise _NotSupported(op)


def _build_jax_body(model: onnx.ModelProto) -> tuple[list[str], list[str], bool]:
    """
    Returns (weight_lines, fn_body_lines, ok).
    ok=False if any op is unsupported (fall back to dispatch).
    """
    init_map: dict[str, np.ndarray] = {
        i.name: numpy_helper.to_array(i).copy()
        for i in model.graph.initializer
    }

    weight_lines: list[str] = []
    var_map: dict[str, str] = {}   # onnx name → python var name

    # ── Embed weights as hex numpy arrays ─────────────────────────────────────
    _INT_DTYPES = {np.dtype('int32'), np.dtype('int64'), np.dtype('uint8'), np.dtype('bool')}
    for idx, (name, arr) in enumerate(init_map.items()):
        vname = f"_w{idx}"
        var_map[name] = vname
        if arr.dtype in _INT_DTYPES or np.issubdtype(arr.dtype, np.integer):
            # Integer weights (shapes, indices): keep as int64
            arr_out = np.ascontiguousarray(arr.astype(np.int64))
            hex_str = arr_out.tobytes().hex()
            weight_lines.append(
                f'{vname} = np.frombuffer(bytes.fromhex("{hex_str}"), '
                f'dtype=np.int64).reshape({list(arr_out.shape)})'
            )
        else:
            arr_f32 = np.ascontiguousarray(arr.astype(np.float32))
            hex_str = arr_f32.tobytes().hex()
            weight_lines.append(
                f'{vname} = np.frombuffer(bytes.fromhex("{hex_str}"), '
                f'dtype=np.float32).reshape({list(arr_f32.shape)})'
            )

    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    var_map[inp_name] = "x"

    shapes: dict[str, tuple] = {}   # not used for jax gen but kept for future use

    fn_body: list[str] = []

    try:
        for node_idx, node in enumerate(model.graph.node):
            op = node.op_type
            ins = [var_map.get(n, _vname(n)) if n else None for n in node.input]

            if op == "Split":
                # multi-output: split → list, then unpack
                x = ins[0]
                axis = int(_attr(node, "axis", 0))
                split_t = ins[1] if len(ins) > 1 and ins[1] else None
                sizes = _attr(node, "split", None)
                if sizes is None and split_t:
                    # static split sizes from initializer
                    sizes = init_map.get(node.input[1], None)
                    if sizes is not None:
                        sizes = sizes.tolist()
                if sizes is None:
                    n = len([o for o in node.output if o])
                    if n == 0: continue
                    # equal split — need shape info
                    raise _NotSupported("Split without static sizes")
                indices = [int(s) for s in np.cumsum(sizes[:-1])]
                tmp = f"_split_{node_idx}"
                fn_body.append(f"{tmp} = jnp.split({x}, {indices}, axis={axis})")
                for k, out_name_k in enumerate(node.output):
                    if out_name_k:
                        vk = _vname(out_name_k)
                        fn_body.append(f"{vk} = {tmp}[{k}]")
                        var_map[out_name_k] = vk
                continue

            expr = _gen_op(op, node, ins, shapes, {})
            if len(node.output) == 1 and node.output[0]:
                vout = _vname(node.output[0])
                fn_body.append(f"{vout} = {expr}")
                var_map[node.output[0]] = vout
            elif len(node.output) > 1:
                for k, out_name_k in enumerate(node.output):
                    if out_name_k:
                        vk = _vname(out_name_k)
                        fn_body.append(f"{vk} = ({expr})[{k}]")
                        var_map[out_name_k] = vk

    except _NotSupported:
        return weight_lines, fn_body, False

    final_var = var_map.get(model.graph.output[0].name, _vname(model.graph.output[0].name))
    fn_body.append(f"return jnp.asarray({final_var}, dtype=jnp.float32)")
    return weight_lines, fn_body, True


# ── Repro generators ──────────────────────────────────────────────────────────

def _inp_line(name: str, arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    return (f'INPUT = np.frombuffer(bytes.fromhex("{arr.tobytes().hex()}"), '
            f'dtype=np.float32).reshape({list(arr.shape)})  '
            f'# input "{name}"')


def make_jax_repro_native(uid: int, model: onnx.ModelProto, inp: np.ndarray,
                           patterns: list, inp_name: str,
                           weight_lines: list[str], fn_body: list[str],
                           expected: Optional[np.ndarray] = None) -> str:
    fn_src = "\n".join(f"    {l}" for l in fn_body)
    weights = "\n".join(weight_lines)
    inp_line = _inp_line(inp_name, inp)

    if expected is not None:
        # Bug is impl-diff (xla/tvm vs pytorch_eager): compare jax.jit vs embedded reference
        exp_arr = np.ascontiguousarray(expected.astype(np.float32))
        exp_line = (f'EXPECTED = np.frombuffer(bytes.fromhex("{exp_arr.tobytes().hex()}"), '
                    f'dtype=np.float32)  # pytorch_eager reference')
        bug_desc = "jax.jit (XLA) produces wrong output vs PyTorch eager reference"
        comparison = f'''\
    x_jax = jnp.array(INPUT)
    out = np.array(jax.jit(model)(x_jax), dtype=np.float32).ravel()

    diff = float(np.linalg.norm(EXPECTED.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(EXPECTED.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {{EXPECTED[:6]}}")
    print(f"actual   (jax.jit/XLA) : {{out[:6]}}")
    print(f"rel L2 : {{diff:.4e}}")'''
    else:
        # True JIT compiler bug: jax.jit diverges from jax.disable_jit
        exp_line = ""
        bug_desc = "jax.jit produces wrong output vs jax.disable_jit"
        comparison = f'''\
    x_jax = jnp.array(INPUT)

    with jax.disable_jit():
        ref = np.array(model(x_jax), dtype=np.float32).ravel()

    out = np.array(jax.jit(model)(x_jax), dtype=np.float32).ravel()

    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (jax.disable_jit): {{ref[:6]}}")
    print(f"actual   (jax.jit)        : {{out[:6]}}")
    print(f"rel L2 : {{diff:.4e}}")'''

    exp_section = f"\n{exp_line}\n" if exp_line else ""
    return f'''\
#!/usr/bin/env python3
"""
Bug #{uid:04d}: {bug_desc}.

Patterns    : {patterns}
Dependencies: numpy jax
Run         : python unique_{uid:04d}.py  →  exit 0 = BUG REPRODUCED
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import jax.scipy.special

# ── Model weights ─────────────────────────────────────────────────────────────
{weights}

# ── Triggering input ─────────────────────────────────────────────────────────
{inp_line}
{exp_section}

# ── Model computation (translated from ONNX) ──────────────────────────────────
def model(x):
{fn_src}


if __name__ == "__main__":
{comparison}
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)
'''


def make_ort_repro(uid: int, model_name: str, inp: np.ndarray,
                   patterns: list, inp_name: str) -> str:
    return f'''\
#!/usr/bin/env python3
"""
Bug #{uid:04d}: OnnxRuntime graph optimizer produces wrong output.

Patterns    : {patterns}
Dependencies: numpy onnx onnxruntime
Run         : python unique_{uid:04d}.py  →  exit 0 = BUG REPRODUCED
"""
import os, sys
import numpy as np
import onnx, onnxruntime as ort

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{model_name}")
{_inp_line(inp_name, inp)}


def reference():
    """ORT with NO optimisation — ground truth."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
    return sess.run(None, {{"{inp_name}": INPUT}})[0].ravel()


def target():
    """ORT with FULL graph optimisation — produces wrong output."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
    return sess.run(None, {{"{inp_name}": INPUT}})[0].ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (ORT_DISABLE_ALL): {{ref[:6]}}")
    print(f"actual   (ORT_ENABLE_ALL) : {{out[:6]}}")
    print(f"rel L2 : {{diff:.4e}}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)
'''


_TC_PATCH = '''\
def _patch():
    try:
        import onnx2torch.node_converters.reshape as _r
        import onnx2torch.node_converters.constant_of_shape as _c
        import torch
        @staticmethod
        def _safe_reshape(inp, shape):
            return inp.reshape([shape[i] for i in range(shape.shape[0])])
        _r.OnnxReshape._do_reshape = _safe_reshape
        def _safe_cos(self, shape):
            v = self.value.item()
            return torch.full([shape[i] for i in range(shape.shape[0])],
                              fill_value=int(v) if isinstance(v, bool) else v,
                              dtype=self.value.dtype, device=self.value.device)
        _c.OnnxConstantOfShape.forward = _safe_cos
    except Exception:
        pass
_patch()
'''


def make_tc_repro(uid: int, model_name: str, inp: np.ndarray,
                  patterns: list, inp_name: str) -> str:
    return f'''\
#!/usr/bin/env python3
"""
Bug #{uid:04d}: torch.compile(inductor) produces wrong output.

Patterns    : {patterns}
Dependencies: numpy onnx torch onnx2torch
Run         : python unique_{uid:04d}.py  →  exit 0 = BUG REPRODUCED
"""
import os, sys, concurrent.futures
import numpy as np
import onnx, onnx2torch
import torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{model_name}")
{_inp_line(inp_name, inp)}

{_TC_PATCH}

if __name__ == "__main__":
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(device)
    x = torch.from_numpy(INPUT).to(device)

    with torch.no_grad():
        ref = m(x).cpu().numpy().ravel()

    compiled = torch.compile(m, backend="inductor", mode="default", fullgraph=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        out = ex.submit(lambda: compiled(x)).result(timeout=90)
    out = out.detach().cpu().numpy().ravel()

    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (eager)         : {{ref[:6]}}")
    print(f"actual   (torch.compile) : {{out[:6]}}")
    print(f"rel L2 : {{diff:.4e}}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)
'''


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="full_campaign")
    ap.add_argument("--manifest", default="full_campaign/bugs_final/manifest.json")
    ap.add_argument("--verify",   default="full_campaign/bugs_final/verify_results.json")
    ap.add_argument("--out",      default="verified_bugs")
    ap.add_argument("--uid-offset", type=int, default=0,
                    help="Add this offset to all unique IDs (avoids collisions when "
                         "appending to an existing verified_bugs/ directory)")
    args = ap.parse_args()

    camp = Path(args.campaign)
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Auto-detect offset if not specified: use max existing UID + 1
    uid_offset = args.uid_offset
    if uid_offset == 0:
        existing = sorted(int(f.stem.replace("unique_",""))
                          for f in out.glob("unique_*.py"))
        if existing:
            uid_offset = existing[-1] + 1

    manifest = json.loads(Path(args.manifest).read_text())
    verify   = json.loads(Path(args.verify).read_text())
    genuine  = {f"unique_{e['unique_id']:04d}.py": e for e in manifest["entries"]}

    written = 0
    for fname, rc in sorted(verify.items()):
        if rc != 0:
            continue
        uid      = int(fname.replace("unique_","").replace(".py","")) + uid_offset
        entry    = genuine[fname]
        src_bug  = entry["source_bug"]
        backends = entry["genuine_backends"]

        src_onnx = camp / f"{src_bug}.onnx"
        report   = json.loads((camp / f"{src_bug}_report.json").read_text())
        model    = onnx.load(str(src_onnx))

        inp_name = model.graph.input[0].name
        inp_flat = report["bug_inputs"].get(inp_name, [])
        shape    = [d.dim_value or 1
                    for d in model.graph.input[0].type.tensor_type.shape.dim]
        inp      = np.asarray(inp_flat, dtype=np.float32).reshape(shape)
        patterns = report.get("pattern_sequence", [])

        is_jax = "xla" in backends or "tvm" in backends
        is_tc  = "torch_compile" in backends and not is_jax

        model_name = f"unique_{uid:04d}.onnx"

        if is_jax:
            weight_lines, fn_body, ok = _build_jax_body(model)
            if ok:
                # Determine if this is a true JIT bug or impl-diff bug
                delta = report.get("delta_opt", {})
                xla_delta = max(float(delta.get("xla", 0)), float(delta.get("tvm", 0)))
                if xla_delta > 0.001:
                    # True JIT compiler bug: jax.jit vs jax.disable_jit
                    expected_arr = None
                else:
                    # Impl-diff bug: XLA/TVM vs pytorch_eager — embed reference output
                    ref_flat = report.get("expected_outputs", {}).get("__reference__", [])
                    expected_arr = np.asarray(ref_flat, dtype=np.float32).ravel() if ref_flat else None
                src  = make_jax_repro_native(uid, model, inp, patterns,
                                             inp_name, weight_lines, fn_body,
                                             expected=expected_arr)
                note = "native-jax"
            else:
                # Can't generate native JAX — skip (ORT fallback won't reproduce XLA bugs)
                ops = sorted({n.op_type for n in model.graph.node})
                print(f"unique_{uid:04d}: [SKIP unsupported-ops for jax]  ops={ops}")
                continue
        elif is_tc:
            shutil.copy2(str(src_onnx), out / model_name)
            src  = make_tc_repro(uid, model_name, inp, patterns, inp_name)
            note = "torch-compile"
        else:
            shutil.copy2(str(src_onnx), out / model_name)
            src  = make_ort_repro(uid, model_name, inp, patterns, inp_name)
            note = "ort"

        py_path = out / f"unique_{uid:04d}.py"
        py_path.write_text(src)

        # ── Verify the clean repro actually reproduces the bug ────────────────
        try:
            result = subprocess.run(
                [sys.executable, str(py_path)],
                capture_output=True, timeout=120,
            )
            verified = (result.returncode == 0)
        except subprocess.TimeoutExpired:
            verified = False
            result = None

        if verified:
            written += 1
            ops = sorted({n.op_type for n in model.graph.node})
            print(f"unique_{uid:04d}: [VERIFIED {note}]  backends={backends}  ops={ops}")
        else:
            # Clean repro didn't reproduce — remove the file (and any .onnx)
            py_path.unlink(missing_ok=True)
            onnx_copy = out / model_name
            if onnx_copy.exists():
                onnx_copy.unlink()
            stderr = (result.stderr.decode(errors='replace')[:200]
                      if result else "timeout")
            print(f"unique_{uid:04d}: [FAILED verify {note}]  {stderr!r}")

    print(f"\nVerified: {written} bugs → {out}/")


if __name__ == "__main__":
    main()
