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
        # Broadcast a [1, C, 1, 1] channel-wise scale to [N, C, H, W], then add
        # shift.  Using non-trivial values so the output is never trivially zero.
        scale = np.full((1, C, 1, 1), 0.9, dtype=np.float32)
        shift = np.full((1, C, 1, 1), 0.1, dtype=np.float32)
        target_shape = np.array([N, C, H, W], dtype=np.int64)

        exp_o = f"{p}_exp"; mul_o = f"{p}_mul"; out = f"{p}_out"
        nodes = [
            helper.make_node("Expand", [f"{p}_scale", f"{p}_tshape"], [exp_o]),
            helper.make_node("Mul",    [input_name, exp_o], [mul_o]),
            helper.make_node("Add",    [mul_o, f"{p}_shift"], [out]),
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
        denom = np.full(ctx.shape, 1.1, dtype=np.float32)

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
        min_v = np.array(-5.5, dtype=np.float32)
        max_v = np.array(5.5, dtype=np.float32)
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
        mean  = np.zeros(ctx.shape, dtype=np.float32)
        scale = np.full(ctx.shape, 1.01, dtype=np.float32)
        bias  = np.zeros(ctx.shape, dtype=np.float32)

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
            scale = np.ones((1, C, 1, 1), dtype=np.float32)
            bias  = np.zeros((1, C, 1, 1), dtype=np.float32)
        else:
            scale = np.ones(ctx.shape, dtype=np.float32)
            bias  = np.zeros(ctx.shape, dtype=np.float32)

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
        eps  = np.array([1e-8], dtype=np.float32)
        # opset 13+: ReduceSum takes axes as a 1-D int64 input, not attribute
        axes = np.array([-1], dtype=np.int64)

        sq_o = f"{p}_sq"; sum_o = f"{p}_sum"; add_o = f"{p}_add"
        sqrt_o = f"{p}_sqrt"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul",       [input_name, input_name], [sq_o]),
            helper.make_node("ReduceSum", [sq_o, f"{p}_axes"], [sum_o], keepdims=1),
            helper.make_node("Add",       [sum_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",      [add_o], [sqrt_o]),
            helper.make_node("Div",       [input_name, sqrt_o], [out]),
        ]
        inits = [numpy_helper.from_array(eps,  f"{p}_eps"),
                 numpy_helper.from_array(axes, f"{p}_axes")]
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
        lo   = np.array([-1.75], dtype=np.float32)
        hi   = np.array([1.75], dtype=np.float32)
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


# ── 11. Where with variable mask ──────────────────────────────────────────
class WhereMask(OTP):
    """Where(x > threshold, x, 0): conditional selection based on value."""
    name = "where_mask"
    category = CAT_BROADCAST
    target_optimization = "where_conditional_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "wm")
        threshold = np.array([0.0], dtype=np.float32)
        zeros = np.zeros(ctx.shape, dtype=np.float32)

        gt_o = f"{p}_gt"; out = f"{p}_out"
        nodes = [
            helper.make_node("Greater", [input_name, f"{p}_thresh"], [gt_o]),
            helper.make_node("Where",   [gt_o, input_name, f"{p}_zeros"], [out]),
        ]
        inits = [numpy_helper.from_array(threshold, f"{p}_thresh"),
                 numpy_helper.from_array(zeros,     f"{p}_zeros")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 12. CumSum (prefix sum) ───────────────────────────────────────────────
class CumSumOp(OTP):
    """CumSum along last axis. Scan-like op; rarely compiler-optimized correctly."""
    name = "cumsum"
    category = CAT_BROADCAST
    target_optimization = "cumsum_loop_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 1

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "cum")
        axis = np.array(ctx.rank - 1, dtype=np.int64)
        scale = np.ones(ctx.shape, dtype=np.float32)

        cs_o = f"{p}_cs"; out = f"{p}_out"
        nodes = [
            helper.make_node("CumSum", [input_name, f"{p}_axis"], [cs_o],
                             exclusive=0, reverse=0),
            helper.make_node("Mul",    [cs_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(axis,  f"{p}_axis"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 13. ReduceMax → Sub → Exp (LogSumExp numerator) ──────────────────────
class LogSumExpStep(OTP):
    """ReduceMax + Sub + Exp: log-sum-exp numerically stable step."""
    name = "logsumexp_step"
    category = CAT_BROADCAST
    target_optimization = "logsumexp_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "lse")
        scale = np.ones(ctx.shape, dtype=np.float32)

        mx_o = f"{p}_mx"; sub_o = f"{p}_sub"; exp_o = f"{p}_exp"
        sum_o = f"{p}_sum"; log_o = f"{p}_log"; out = f"{p}_out"
        axes = np.array([-1], dtype=np.int64)
        nodes = [
            helper.make_node("ReduceMax", [input_name], [mx_o],
                             axes=[-1], keepdims=1),
            helper.make_node("Sub",       [input_name, mx_o], [sub_o]),
            helper.make_node("Exp",       [sub_o], [exp_o]),
            helper.make_node("ReduceSum", [exp_o, f"{p}_axes"], [sum_o], keepdims=1),
            helper.make_node("Log",       [sum_o], [log_o]),
            helper.make_node("Add",       [log_o, mx_o], [out]),
        ]
        return PatternInstance(self.name, self.category, nodes,
                               [numpy_helper.from_array(axes, f"{p}_axes")],
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape[:-1]) + [1],
                                                 ctx.dtype, ctx.layout))


# ── 14. Abs → Neg → Relu (piecewise rectification) ───────────────────────
class AbsNegReLU(OTP):
    """Neg + Abs + ReLU pattern.

    Computes ReLU(Abs(Neg(x))) * alpha = Abs(x) * alpha.
    Tests piecewise arithmetic rewriting (abs/relu interaction).
    Note: order is Neg → Abs → ReLU (not Abs → Neg → ReLU which
    always outputs zero since Abs gives ≥0, Neg gives ≤0, ReLU clips to 0).
    """
    name = "abs_neg_relu"
    category = CAT_BROADCAST
    target_optimization = "abs_arithmetic_simplification"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "anr")
        alpha = np.full(ctx.shape, 0.5, dtype=np.float32)

        neg_o = f"{p}_neg"; abs_o = f"{p}_abs"; relu_o = f"{p}_relu"; out = f"{p}_out"
        nodes = [
            helper.make_node("Neg",  [input_name], [neg_o]),
            helper.make_node("Abs",  [neg_o], [abs_o]),
            helper.make_node("Relu", [abs_o], [relu_o]),
            helper.make_node("Mul",  [relu_o, f"{p}_alpha"], [out]),
        ]
        inits = [numpy_helper.from_array(alpha, f"{p}_alpha")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 15. Pow(2) → ReduceSum → Sqrt (Euclidean norm) ───────────────────────
class EuclideanNormBroadcast(OTP):
    """Pow(x,2) + ReduceSum + Sqrt: L2 distance step with broadcasting."""
    name = "euclidean_norm_broadcast"
    category = CAT_BROADCAST
    target_optimization = "norm_broadcast_simplification"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "enb")
        eps  = np.array([1e-8], dtype=np.float32)
        bias = np.zeros(ctx.shape, dtype=np.float32)
        axes = np.array([-1], dtype=np.int64)

        diff_o = f"{p}_diff"; sq_o = f"{p}_sq"
        sm_o = f"{p}_sm"; add_o = f"{p}_add"; sqrt_o = f"{p}_sqrt"; out = f"{p}_out"
        nodes = [
            helper.make_node("Sub",       [input_name, f"{p}_bias"], [diff_o]),
            helper.make_node("Mul",       [diff_o, diff_o], [sq_o]),
            helper.make_node("ReduceSum", [sq_o, f"{p}_axes"], [sm_o], keepdims=1),
            helper.make_node("Add",       [sm_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",      [add_o], [sqrt_o]),
            helper.make_node("Div",       [diff_o, sqrt_o], [out]),
        ]
        inits = [numpy_helper.from_array(eps,  f"{p}_eps"),
                 numpy_helper.from_array(bias, f"{p}_bias"),
                 numpy_helper.from_array(axes, f"{p}_axes")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 16. Softmax → TopK Proxy (Softmax + Mul by one-hot) ──────────────────
class SoftmaxMul(OTP):
    """Softmax then element-wise Mul: attention weighting / scoring."""
    name = "softmax_mul"
    category = CAT_BROADCAST
    target_optimization = "softmax_mul_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "sftm")
        temperature = np.array([1.25], dtype=np.float32)
        weight = np.ones(ctx.shape, dtype=np.float32)

        div_o = f"{p}_div"; sm_o = f"{p}_sm"; out = f"{p}_out"
        nodes = [
            helper.make_node("Div",     [input_name, f"{p}_temp"], [div_o]),
            helper.make_node("Softmax", [div_o], [sm_o], axis=-1),
            helper.make_node("Mul",     [sm_o, f"{p}_weight"], [out]),
        ]
        inits = [numpy_helper.from_array(temperature, f"{p}_temp"),
                 numpy_helper.from_array(weight,      f"{p}_weight")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 17. Min+Max clipped affine ────────────────────────────────────────────
class ClippedAffine(OTP):
    """Affine(x) then double-sided clip. Tests Min/Max chain optimization."""
    name = "clipped_affine"
    category = CAT_BROADCAST
    target_optimization = "clipped_affine_canonicalization"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "caff")
        scale = np.array([1.25], dtype=np.float32)
        shift = np.array([0.0], dtype=np.float32)
        lo = np.array([-1.0], dtype=np.float32)
        hi = np.array([1.0],  dtype=np.float32)

        sc_o = f"{p}_sc"; add_o = f"{p}_add"
        mx_o = f"{p}_mx"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul",  [input_name, f"{p}_scale"], [sc_o]),
            helper.make_node("Add",  [sc_o, f"{p}_shift"], [add_o]),
            helper.make_node("Max",  [add_o, f"{p}_lo"], [mx_o]),
            helper.make_node("Min",  [mx_o, f"{p}_hi"], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_scale"),
                 numpy_helper.from_array(shift, f"{p}_shift"),
                 numpy_helper.from_array(lo,    f"{p}_lo"),
                 numpy_helper.from_array(hi,    f"{p}_hi")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 18. Floor → Ceil → Round chain ───────────────────────────────────────
class FloorCeilRound(OTP):
    """Floor + Ceil + Round sequence: integer-rounding canonicalization."""
    name = "floor_ceil_round"
    category = CAT_BROADCAST
    target_optimization = "rounding_canonicalization"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "fcr")
        # scale by 3.7 so values span several integers → floor/ceil differ per element
        scale = np.full(ctx.shape, 3.7, dtype=np.float32)
        half  = np.full(ctx.shape, 0.5, dtype=np.float32)

        sc_o = f"{p}_sc"; fl_o = f"{p}_fl"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul",   [input_name, f"{p}_scale"], [sc_o]),
            helper.make_node("Floor", [sc_o],  [fl_o]),
            helper.make_node("Add",   [fl_o, f"{p}_half"], [add_o]),
            helper.make_node("Round", [add_o], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_scale"),
                 numpy_helper.from_array(half,  f"{p}_half")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 19. GELU approximation (tanh-based) ──────────────────────────────────
class GeluApprox(OTP):
    """Tanh-based GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π)*(x + 0.044715*x³))).
    Tests non-linear composition optimization."""
    name = "gelu_approx"
    category = CAT_BROADCAST
    target_optimization = "gelu_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "gelu")
        c1    = np.array([0.044715], dtype=np.float32)
        c2    = np.array([np.sqrt(2.0 / np.pi)], dtype=np.float32)
        one   = np.array([1.0], dtype=np.float32)
        half  = np.array([0.5], dtype=np.float32)

        x3_o  = f"{p}_x3";  sc_o  = f"{p}_sc"
        inn_o = f"{p}_inn"; mlt_o = f"{p}_mlt"
        th_o  = f"{p}_th";  add_o = f"{p}_add"
        hf_o  = f"{p}_hf";  out   = f"{p}_out"
        # x^3 via Mul-chain: Pow(x, 3.0) on negative base is implementation-
        # defined (some backends lower via exp/log → NaN). Mul is unambiguous.
        x2_o = f"{p}_x2"
        nodes = [
            helper.make_node("Mul",  [input_name, input_name], [x2_o]),
            helper.make_node("Mul",  [x2_o, input_name], [x3_o]),
            helper.make_node("Mul",  [x3_o, f"{p}_c1"], [sc_o]),
            helper.make_node("Add",  [input_name, sc_o], [inn_o]),
            helper.make_node("Mul",  [inn_o, f"{p}_c2"], [mlt_o]),
            helper.make_node("Tanh", [mlt_o], [th_o]),
            helper.make_node("Add",  [th_o, f"{p}_one"], [add_o]),
            helper.make_node("Mul",  [input_name, f"{p}_half"], [hf_o]),
            helper.make_node("Mul",  [hf_o, add_o], [out]),
        ]
        inits = [numpy_helper.from_array(c1,    f"{p}_c1"),
                 numpy_helper.from_array(c2,    f"{p}_c2"),
                 numpy_helper.from_array(one,   f"{p}_one"),
                 numpy_helper.from_array(half,  f"{p}_half")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 20. Mish activation: x * tanh(softplus(x)) ───────────────────────────
class MishActivation(OTP):
    """Mish: x * tanh(log(1+exp(x))). Tests chained non-linear fusions."""
    name = "mish_activation"
    category = CAT_BROADCAST
    target_optimization = "mish_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "mish")
        one = np.ones(ctx.shape, dtype=np.float32)

        exp_o = f"{p}_exp"; sp_o = f"{p}_sp"
        log_o = f"{p}_log"; th_o = f"{p}_th"; out = f"{p}_out"
        nodes = [
            helper.make_node("Exp",  [input_name], [exp_o]),
            helper.make_node("Add",  [exp_o, f"{p}_one"], [sp_o]),
            helper.make_node("Log",  [sp_o], [log_o]),
            helper.make_node("Tanh", [log_o], [th_o]),
            helper.make_node("Mul",  [input_name, th_o], [out]),
        ]
        inits = [numpy_helper.from_array(one, f"{p}_one")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 21. HardSwish: x * ReLU6(x+3) / 6 ───────────────────────────────────
class HardSwish(OTP):
    """HardSwish activation. Tests min/max/clip fused with mul."""
    name = "hard_swish"
    category = CAT_BROADCAST
    target_optimization = "hardswish_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "hsw")
        three = np.array([3.0], dtype=np.float32)
        six   = np.array([6.0], dtype=np.float32)
        lo    = np.array([0.0], dtype=np.float32)
        hi    = np.array([6.0], dtype=np.float32)

        add_o = f"{p}_add"; clip_o = f"{p}_clip"
        div_o = f"{p}_div"; out = f"{p}_out"
        nodes = [
            helper.make_node("Add",  [input_name, f"{p}_three"], [add_o]),
            helper.make_node("Clip", [add_o, f"{p}_lo", f"{p}_hi"], [clip_o]),
            helper.make_node("Div",  [clip_o, f"{p}_six"], [div_o]),
            helper.make_node("Mul",  [input_name, div_o], [out]),
        ]
        inits = [numpy_helper.from_array(three, f"{p}_three"),
                 numpy_helper.from_array(six,   f"{p}_six"),
                 numpy_helper.from_array(lo,    f"{p}_lo"),
                 numpy_helper.from_array(hi,    f"{p}_hi")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 22. Variance normalization ────────────────────────────────────────────
class VarianceNorm(OTP):
    """x / (std(x) + eps): tests variance + sqrt + reciprocal chain."""
    name = "variance_norm"
    category = CAT_BROADCAST
    target_optimization = "variance_normalization"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "vn")
        eps  = np.array([1e-6], dtype=np.float32)
        axes = np.array([-1], dtype=np.int64)

        sq_o = f"{p}_sq"; sm_o = f"{p}_sm"
        add_o = f"{p}_add"; sqrt_o = f"{p}_sqrt"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul",       [input_name, input_name], [sq_o]),
            helper.make_node("ReduceSum", [sq_o, f"{p}_axes"], [sm_o], keepdims=1),
            helper.make_node("Add",       [sm_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",      [add_o], [sqrt_o]),
            helper.make_node("Div",       [input_name, sqrt_o], [out]),
        ]
        inits = [numpy_helper.from_array(eps,  f"{p}_eps"),
                 numpy_helper.from_array(axes, f"{p}_axes")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 23. PReLU (learnable negative slope) ─────────────────────────────────
class PReLU(OTP):
    """PReLU: max(0,x) + slope*min(0,x). Tests per-channel slope fusion."""
    name = "prelu"
    category = CAT_BROADCAST
    target_optimization = "prelu_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "prl")
        slope = np.full(ctx.shape, 0.01, dtype=np.float32)
        zero  = np.zeros(ctx.shape, dtype=np.float32)

        pos_o=f"{p}_pos"; neg_o=f"{p}_neg"; sl_o=f"{p}_sl"; out=f"{p}_out"
        nodes = [
            helper.make_node("Relu",  [input_name], [pos_o]),
            helper.make_node("Min",   [input_name, f"{p}_zero"], [neg_o]),
            helper.make_node("Mul",   [neg_o, f"{p}_slope"], [sl_o]),
            helper.make_node("Add",   [pos_o, sl_o], [out]),
        ]
        inits = [numpy_helper.from_array(slope, f"{p}_slope"),
                 numpy_helper.from_array(zero,  f"{p}_zero")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout))


# ── 24. Sigmoid + Scale (binary cross-entropy logit transform) ────────────
class SigmoidScale(OTP):
    """Sigmoid(x) * 2 - 1 → maps to [-1,1]. Tests sigmoid+affine fusion."""
    name = "sigmoid_scale"
    category = CAT_BROADCAST
    target_optimization = "sigmoid_affine_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "sigs")
        two = np.full(ctx.shape, 2.0, dtype=np.float32)
        one = np.full(ctx.shape, 1.0, dtype=np.float32)

        sig_o=f"{p}_sig"; sc_o=f"{p}_sc"; out=f"{p}_out"
        nodes = [
            helper.make_node("Sigmoid", [input_name], [sig_o]),
            helper.make_node("Mul",     [sig_o, f"{p}_two"], [sc_o]),
            helper.make_node("Sub",     [sc_o, f"{p}_one"], [out]),
        ]
        inits = [numpy_helper.from_array(two, f"{p}_two"),
                 numpy_helper.from_array(one, f"{p}_one")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout))


# ── 25. Elu + Scale (EfficientNet activation variant) ─────────────────────
class EluScale(OTP):
    """ELU(x) + 1: shifts ELU to be non-negative. Tests ELU offset fold."""
    name = "elu_scale"
    category = CAT_BROADCAST
    target_optimization = "elu_offset_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "elus")
        one = np.full(ctx.shape, 1.0, dtype=np.float32)

        elu_o=f"{p}_elu"; out=f"{p}_out"
        nodes = [
            helper.make_node("Elu", [input_name], [elu_o], alpha=1.0),
            helper.make_node("Add", [elu_o, f"{p}_one"], [out]),
        ]
        inits = [numpy_helper.from_array(one, f"{p}_one")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout))


# ── 26. Reduce + Broadcast (softmax-like normalization) ───────────────────
class ReduceBroadcastNorm(OTP):
    """x / ReduceSum(x, keepdims): probability simplex projection."""
    name = "reduce_broadcast_norm"
    category = CAT_BROADCAST
    target_optimization = "reduce_broadcast_fusion"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "rbn")
        eps = np.array([1e-7], dtype=np.float32)
        axes = np.array([-1], dtype=np.int64)

        abs_o=f"{p}_abs"; rs_o=f"{p}_rs"; add_o=f"{p}_add"; out=f"{p}_out"
        nodes = [
            helper.make_node("Abs",       [input_name], [abs_o]),
            helper.make_node("ReduceSum", [abs_o, f"{p}_axes"], [rs_o], keepdims=1),
            helper.make_node("Add",       [rs_o, f"{p}_eps"], [add_o]),
            helper.make_node("Div",       [abs_o, add_o], [out]),
        ]
        inits = [numpy_helper.from_array(eps,  f"{p}_eps"),
                 numpy_helper.from_array(axes, f"{p}_axes")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout))


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
    WhereMask(),
    CumSumOp(),
    LogSumExpStep(),
    AbsNegReLU(),
    EuclideanNormBroadcast(),
    SoftmaxMul(),
    ClippedAffine(),
    FloorCeilRound(),
    GeluApprox(),
    MishActivation(),
    HardSwish(),
    VarianceNorm(),
    PReLU(),
    SigmoidScale(),
    EluScale(),
    ReduceBroadcastNorm(),
]
