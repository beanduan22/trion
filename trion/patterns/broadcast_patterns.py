"""
Broadcast- and arithmetic-sensitive OTPs.
Target: broadcast rewrite, arithmetic simplification, constant propagation.
"""
from __future__ import annotations
import numpy as np
from onnx import helper, numpy_helper
from typing import Optional

from .base import OTP, PatternInstance, CAT_BROADCAST
from ..generation.context import StructuralContext


# ── 1. Expand → Add → Mul ─────────────────────────────────────────────────
class ExpandAddMul(OTP):
    name = "expand_add_mul"
    category = CAT_BROADCAST
    target_optimization = "broadcast_rewrite"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "eam")
        # Broadcast a [1, C, 1, 1] tensor to [N, C, H, W]
        scale = rng.normal(1.0, 0.1, (1, C, 1, 1)).astype(np.float32)
        shift = rng.normal(0.0, 0.1, (1, C, 1, 1)).astype(np.float32)
        target_shape = np.array([N, C, H, W], dtype=np.int64)

        exp_o = f"{p}_exp"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Expand", [f"{p}_scale", f"{p}_tshape"], [exp_o]),
            helper.make_node("Add",    [input_name, exp_o], [add_o]),
            helper.make_node("Mul",    [add_o, f"{p}_shift"], [out]),
        ]
        inits = [numpy_helper.from_array(scale,        f"{p}_scale"),
                 numpy_helper.from_array(target_shape, f"{p}_tshape"),
                 numpy_helper.from_array(shift,        f"{p}_shift")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 2. Reciprocal → Mul (x * 1/y → div rewrite) ───────────────────────────
class ReciprocalMul(OTP):
    name = "reciprocal_mul"
    category = CAT_BROADCAST
    target_optimization = "reciprocal_div_rewrite"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "recmul")
        denom = (np.abs(rng.normal(1.0, 0.5, ctx.shape)) + 0.1).astype(np.float32)

        rec_o = f"{p}_rec"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reciprocal", [f"{p}_denom"], [rec_o]),
            helper.make_node("Mul",        [input_name, rec_o], [out]),
        ]
        inits = [numpy_helper.from_array(denom, f"{p}_denom")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 3. Log → Clamp (log + clip for numerical stability) ───────────────────
class LogClamp(OTP):
    name = "log_clamp"
    category = CAT_BROADCAST
    target_optimization = "log_clamp_simplification"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "logclamp")
        # Ensure positive input by abs first
        min_v = np.array(float(rng.uniform(-10, -1)),  dtype=np.float32)
        max_v = np.array(float(rng.uniform(1,  10)),   dtype=np.float32)
        eps   = np.full(ctx.shape, 1e-6, dtype=np.float32)

        abs_o = f"{p}_abs"; add_o = f"{p}_add"; log_o = f"{p}_log"; out = f"{p}_out"
        nodes = [
            helper.make_node("Abs", [input_name], [abs_o]),
            helper.make_node("Add", [abs_o, f"{p}_eps"], [add_o]),
            helper.make_node("Log", [add_o], [log_o]),
            helper.make_node("Clip", [log_o, f"{p}_min", f"{p}_max"], [out]),
        ]
        inits = [numpy_helper.from_array(eps,   f"{p}_eps"),
                 numpy_helper.from_array(min_v, f"{p}_min"),
                 numpy_helper.from_array(max_v, f"{p}_max")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 4. Exp → Div (Softplus-like: log(1+exp(x)) ────────────────────────────
class ExpDiv(OTP):
    name = "exp_div_softplus"
    category = CAT_BROADCAST
    target_optimization = "exp_arithmetic_simplification"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "expdiv")
        one = np.ones(ctx.shape, dtype=np.float32)

        exp_o = f"{p}_exp"; add_o = f"{p}_add"; log_o = f"{p}_log"; out = f"{p}_out"
        nodes = [
            helper.make_node("Exp", [input_name], [exp_o]),
            helper.make_node("Add", [exp_o, f"{p}_one"], [add_o]),
            helper.make_node("Log", [add_o], [log_o]),
            helper.make_node("Relu", [log_o], [out]),   # softplus is always ≥ 0
        ]
        inits = [numpy_helper.from_array(one, f"{p}_one")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 5. Sub → Mul → Add (affine: a*(x-m)+b) ───────────────────────────────
class SubMulAdd(OTP):
    name = "sub_mul_add"
    category = CAT_BROADCAST
    target_optimization = "affine_transform_simplification"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "sma")
        mean  = rng.normal(0, 0.5, ctx.shape).astype(np.float32)
        scale = (np.abs(rng.normal(1, 0.2, ctx.shape)) + 0.01).astype(np.float32)
        bias  = rng.normal(0, 0.1, ctx.shape).astype(np.float32)

        sub_o = f"{p}_sub"; mul_o = f"{p}_mul"; out = f"{p}_out"
        nodes = [
            helper.make_node("Sub", [input_name, f"{p}_mean"],  [sub_o]),
            helper.make_node("Mul", [sub_o,       f"{p}_scale"], [mul_o]),
            helper.make_node("Add", [mul_o,       f"{p}_bias"],  [out]),
        ]
        inits = [numpy_helper.from_array(mean,  f"{p}_mean"),
                 numpy_helper.from_array(scale, f"{p}_scale"),
                 numpy_helper.from_array(bias,  f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 6. Mul → Add → ReLU (fused scale-shift-activate) ─────────────────────
class MulAddReLU(OTP):
    name = "mul_add_relu"
    category = CAT_BROADCAST
    target_optimization = "scale_shift_fuse"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "mar")
        # Channel-wise scale/shift for 4-D, element-wise for 2-D
        if ctx.rank == 4:
            _, C, _, _ = ctx.shape
            scale = rng.normal(1, 0.1, (1, C, 1, 1)).astype(np.float32)
            bias  = rng.normal(0, 0.1, (1, C, 1, 1)).astype(np.float32)
        else:
            scale = rng.normal(1, 0.1, ctx.shape).astype(np.float32)
            bias  = rng.normal(0, 0.1, ctx.shape).astype(np.float32)

        mul_o = f"{p}_mul"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul", [input_name,  f"{p}_scale"], [mul_o]),
            helper.make_node("Add", [mul_o,        f"{p}_bias"],  [add_o]),
            helper.make_node("Relu", [add_o], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_scale"),
                 numpy_helper.from_array(bias,  f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 7. Sqrt → Div (RMS normalization component) ───────────────────────────
class SqrtDiv(OTP):
    name = "sqrt_div_rms"
    category = CAT_BROADCAST
    target_optimization = "rms_norm_component"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "sqrtdiv")
        eps = np.array([1e-6], dtype=np.float32)

        sq_o = f"{p}_sq"; mean_o = f"{p}_mean"; add_o = f"{p}_add"
        sqrt_o = f"{p}_sqrt"; out = f"{p}_out"
        axes = list(range(1, ctx.rank))
        nodes = [
            helper.make_node("Mul",        [input_name, input_name], [sq_o]),
            helper.make_node("ReduceMean", [sq_o], [mean_o],
                             axes=axes, keepdims=1),
            helper.make_node("Add",        [mean_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [sqrt_o]),
            helper.make_node("Div",        [input_name, sqrt_o], [out]),
        ]
        inits = [numpy_helper.from_array(eps, f"{p}_eps")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 8. Pow → ReduceSum → Sqrt (L2 norm) ──────────────────────────────────
class L2Norm(OTP):
    name = "l2_norm"
    category = CAT_BROADCAST
    target_optimization = "l2_norm_simplification"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "l2n")
        two = np.array([2.0], dtype=np.float32)
        eps = np.array([1e-8], dtype=np.float32)
        axes = [-1]

        sq_o = f"{p}_sq"; sum_o = f"{p}_sum"; add_o = f"{p}_add"
        sqrt_o = f"{p}_sqrt"; out = f"{p}_out"
        nodes = [
            helper.make_node("Pow",        [input_name, f"{p}_two"], [sq_o]),
            helper.make_node("ReduceSum",  [sq_o], [sum_o],
                             axes=axes, keepdims=1),
            helper.make_node("Add",        [sum_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [sqrt_o]),
            helper.make_node("Div",        [input_name, sqrt_o], [out]),
        ]
        inits = [numpy_helper.from_array(two, f"{p}_two"),
                 numpy_helper.from_array(eps, f"{p}_eps")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 9. Min → Max → Sub (hard clamp then range-normalize) ──────────────────
class HardClampNorm(OTP):
    name = "hard_clamp_norm"
    category = CAT_BROADCAST
    target_optimization = "clamp_arithmetic_constant_prop"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "hcn")
        lo   = np.array([float(rng.uniform(-3, -0.5))], dtype=np.float32)
        hi   = np.array([float(rng.uniform(0.5, 3))],   dtype=np.float32)
        denom = np.array([float(hi - lo)], dtype=np.float32)

        cl_o = f"{p}_cl"; sub_o = f"{p}_sub"; out = f"{p}_out"
        nodes = [
            helper.make_node("Clip", [input_name, f"{p}_lo", f"{p}_hi"], [cl_o]),
            helper.make_node("Sub",  [cl_o, f"{p}_lo"], [sub_o]),
            helper.make_node("Div",  [sub_o, f"{p}_denom"], [out]),
        ]
        inits = [numpy_helper.from_array(lo,    f"{p}_lo"),
                 numpy_helper.from_array(hi,    f"{p}_hi"),
                 numpy_helper.from_array(denom, f"{p}_denom")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 10. Sigmoid → Sub → Mul (Swish: x * sigmoid(x)) ──────────────────────
class Swish(OTP):
    name = "swish"
    category = CAT_BROADCAST
    target_optimization = "swish_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "swish")
        sig_o = f"{p}_sig"; out = f"{p}_out"
        nodes = [
            helper.make_node("Sigmoid", [input_name], [sig_o]),
            helper.make_node("Mul",     [input_name, sig_o], [out]),
        ]
        return PatternInstance(self.name, self.category, nodes, [],
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


ALL_BROADCAST_PATTERNS = [
    ExpandAddMul(),
    ReciprocalMul(),
    LogClamp(),
    ExpDiv(),
    SubMulAdd(),
    MulAddReLU(),
    SqrtDiv(),
    L2Norm(),
    HardClampNorm(),
    Swish(),
]
