"""
Normalization- and reduction-sensitive OTPs.
Target: reduction handling, normalization lowering, precision sensitivity.
"""
from __future__ import annotations
import numpy as np
from onnx import helper, numpy_helper
from typing import Optional

from .base import OTP, PatternInstance, CAT_NORMALIZATION
from ..generation.context import StructuralContext


# ── 1. ReduceMean → Sub → Div (manual LayerNorm) ──────────────────────────
class ManualLayerNorm(OTP):
    name = "manual_layernorm"
    category = CAT_NORMALIZATION
    target_optimization = "layernorm_lowering"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "mln")
        eps   = np.array([1e-5], dtype=np.float32)
        axes  = [-1]

        m_o = f"{p}_m"; sub_o = f"{p}_sub"
        sq_o = f"{p}_sq"; var_o = f"{p}_var"
        add_o = f"{p}_add"; sqrt_o = f"{p}_sqrt"; out = f"{p}_out"

        nodes = [
            helper.make_node("ReduceMean", [input_name], [m_o],
                             axes=axes, keepdims=1),
            helper.make_node("Sub",        [input_name, m_o], [sub_o]),
            helper.make_node("Mul",        [sub_o, sub_o], [sq_o]),
            helper.make_node("ReduceMean", [sq_o], [var_o],
                             axes=axes, keepdims=1),
            helper.make_node("Add",        [var_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [sqrt_o]),
            helper.make_node("Div",        [sub_o, sqrt_o], [out]),
        ]
        inits = [numpy_helper.from_array(eps, f"{p}_eps")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 2. LayerNorm → ReLU ────────────────────────────────────────────────────
class LayerNormReLU(OTP):
    name = "layernorm_relu"
    category = CAT_NORMALIZATION
    target_optimization = "layernorm_activation_fusion"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "lnr")
        norm_dim = ctx.shape[-1]
        scale = np.ones(norm_dim, dtype=np.float32)
        bias  = np.zeros(norm_dim, dtype=np.float32)

        ln_o = f"{p}_ln"; out = f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_sc", f"{p}_b"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Relu", [ln_o], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 3. LayerNorm → Add (pre-norm residual) ────────────────────────────────
class LayerNormResidualAdd(OTP):
    name = "layernorm_residual_add"
    category = CAT_NORMALIZATION
    target_optimization = "pre_norm_residual"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "lnradd")
        norm_dim = ctx.shape[-1]
        scale = np.ones(norm_dim, dtype=np.float32)
        bias  = np.zeros(norm_dim, dtype=np.float32)
        res   = np.zeros(ctx.shape, dtype=np.float32)

        ln_o = f"{p}_ln"; out = f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_sc", f"{p}_b"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Add", [ln_o, f"{p}_res"], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b"),
                 numpy_helper.from_array(res,   f"{p}_res")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 4. InstanceNorm → ReLU ────────────────────────────────────────────────
class InstanceNormReLU(OTP):
    name = "instancenorm_relu"
    category = CAT_NORMALIZATION
    target_optimization = "instance_norm_lowering"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "inr")
        scale = np.ones(C, dtype=np.float32)
        bias  = np.zeros(C, dtype=np.float32)

        in_o = f"{p}_in"; out = f"{p}_out"
        nodes = [
            helper.make_node("InstanceNormalization",
                             [input_name, f"{p}_sc", f"{p}_b"], [in_o],
                             epsilon=1e-5),
            helper.make_node("Relu", [in_o], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 5. ReduceMean → Sub → Pow → ReduceMean → Sqrt → Div (RMSNorm) ─────────
class RMSNorm(OTP):
    name = "rmsnorm"
    category = CAT_NORMALIZATION
    target_optimization = "rmsnorm_lowering"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "rms")
        eps = np.array([1e-6], dtype=np.float32)
        axes = [-1]

        sq_o = f"{p}_sq"; mean_o = f"{p}_mean"
        add_o = f"{p}_add"; sqrt_o = f"{p}_sqrt"
        sc_o = f"{p}_sc"; out = f"{p}_out"

        norm_dim = ctx.shape[-1]
        g = np.ones(norm_dim, dtype=np.float32)

        nodes = [
            # Use Mul(x,x) instead of Pow(x, 2.0): Pow with float exponent on
            # negative base is undefined; some backends lower it via exp/log
            # and produce NaN, which would be a model bug, not a compiler bug.
            helper.make_node("Mul",        [input_name, input_name], [sq_o]),
            helper.make_node("ReduceMean", [sq_o], [mean_o],
                             axes=axes, keepdims=1),
            helper.make_node("Add",        [mean_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [sqrt_o]),
            helper.make_node("Div",        [input_name, sqrt_o], [sc_o]),
            helper.make_node("Mul",        [sc_o, f"{p}_g"], [out]),
        ]
        inits = [numpy_helper.from_array(eps, f"{p}_eps"),
                 numpy_helper.from_array(g,   f"{p}_g")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 6. GroupNorm → ReLU ────────────────────────────────────────────────────
class GroupNormReLU(OTP):
    name = "groupnorm_relu"
    category = CAT_NORMALIZATION
    target_optimization = "groupnorm_lowering"

    def is_compatible(self, ctx):
        return (ctx.rank == 4 and ctx.dtype == "float32"
                and ctx.channels() is not None and ctx.channels() % 4 == 0)

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        num_groups = min(4, C // 4) or 1
        p = self._p(node_id, "gnr")
        scale = np.ones(C, dtype=np.float32)
        bias  = np.zeros(C, dtype=np.float32)

        # GroupNorm via reshape + LayerNorm + reshape
        s1 = np.array([-1, C // num_groups * H * W], dtype=np.int64)
        s2 = np.array([-1, C, H, W], dtype=np.int64)
        ln_dim = C // num_groups * H * W
        ln_sc = np.ones(ln_dim,  dtype=np.float32)
        ln_b  = np.zeros(ln_dim, dtype=np.float32)

        r1_o = f"{p}_r1"; ln_o = f"{p}_ln"; r2_o = f"{p}_r2"
        sc_o = f"{p}_sc"; out  = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_s1"], [r1_o]),
            helper.make_node("LayerNormalization",
                             [r1_o, f"{p}_lnsc", f"{p}_lnb"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Reshape", [ln_o, f"{p}_s2"], [r2_o]),
            helper.make_node("Mul", [r2_o, f"{p}_scale"], [sc_o]),
            helper.make_node("Add", [sc_o, f"{p}_bias"],  [f"{p}_aff"]),
            helper.make_node("Relu", [f"{p}_aff"], [out]),
        ]
        inits = [numpy_helper.from_array(s1,    f"{p}_s1"),
                 numpy_helper.from_array(s2,    f"{p}_s2"),
                 numpy_helper.from_array(ln_sc, f"{p}_lnsc"),
                 numpy_helper.from_array(ln_b,  f"{p}_lnb"),
                 numpy_helper.from_array(scale.reshape(1,C,1,1), f"{p}_scale"),
                 numpy_helper.from_array(bias.reshape(1,C,1,1),  f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 7. BatchNorm in eval mode (running stats) ─────────────────────────────
class BatchNormEval(OTP):
    name = "batchnorm_eval"
    category = CAT_NORMALIZATION
    target_optimization = "batchnorm_eval_folding"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "bne")
        scale = np.ones(C, dtype=np.float32)
        bias  = np.zeros(C, dtype=np.float32)
        mean  = np.zeros(C, dtype=np.float32)
        var   = np.ones(C, dtype=np.float32)

        out = f"{p}_out"
        nodes = [
            helper.make_node("BatchNormalization",
                             [input_name,
                              f"{p}_sc", f"{p}_b", f"{p}_m", f"{p}_v"],
                             [out], epsilon=1e-5),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b"),
                 numpy_helper.from_array(mean,  f"{p}_m"),
                 numpy_helper.from_array(var,   f"{p}_v")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 8. ReduceMax → Sub → Exp → ReduceSum → Div (stable Softmax) ──────────
class StableSoftmax(OTP):
    name = "stable_softmax"
    category = CAT_NORMALIZATION
    target_optimization = "stable_softmax_rewrite"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "ssm")
        # opset 13+: ReduceSum axes is an input tensor; ReduceMax axes is still attribute
        axes_t = np.array([-1], dtype=np.int64)

        mx_o = f"{p}_mx"; sub_o = f"{p}_sub"; exp_o = f"{p}_exp"
        sum_o = f"{p}_sum"; out = f"{p}_out"
        nodes = [
            helper.make_node("ReduceMax", [input_name], [mx_o],
                             axes=[-1], keepdims=1),
            helper.make_node("Sub",       [input_name, mx_o], [sub_o]),
            helper.make_node("Exp",       [sub_o], [exp_o]),
            helper.make_node("ReduceSum", [exp_o, f"{p}_axes"], [sum_o], keepdims=1),
            helper.make_node("Div",       [exp_o, sum_o], [out]),
        ]
        return PatternInstance(self.name, self.category, nodes,
                               [numpy_helper.from_array(axes_t, f"{p}_axes")],
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 9. ReduceMean along spatial (global context aggregation) ──────────────
class SpatialReduceMean(OTP):
    name = "spatial_reduce_mean"
    category = CAT_NORMALIZATION
    target_optimization = "reduction_lowering"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "srm")

        # Reduce over H, W → [N, C, 1, 1], then broadcast-multiply
        scale = np.ones((1, C, 1, 1), dtype=np.float32)

        red_o = f"{p}_red"; out = f"{p}_out"
        nodes = [
            helper.make_node("ReduceMean", [input_name], [red_o],
                             axes=[2, 3], keepdims=1),
            helper.make_node("Mul", [input_name, red_o], [f"{p}_att"]),
            helper.make_node("Mul", [f"{p}_att", f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 10. LayerNorm → Dropout (inference: dropout = identity) ───────────────
class LayerNormDropout(OTP):
    name = "layernorm_dropout_identity"
    category = CAT_NORMALIZATION
    target_optimization = "dropout_elimination_inference"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "lnd")
        norm_dim = ctx.shape[-1]
        scale = np.ones(norm_dim, dtype=np.float32)
        bias  = np.zeros(norm_dim, dtype=np.float32)
        # Dropout ratio must be a 0-D (scalar) float32 tensor in opset 13+
        ratio = np.float32(0.0)

        ln_o = f"{p}_ln"; out = f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_sc", f"{p}_b"], [ln_o],
                             axis=-1, epsilon=1e-5),
            # Inference dropout (ratio=0) should be eliminated by compilers
            helper.make_node("Dropout", [ln_o, f"{p}_ratio"], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b"),
                 numpy_helper.from_array(ratio, f"{p}_ratio")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 11. Manual GroupNorm (Reshape+LN+Reshape) ─────────────────────────────
class ManualGroupNorm(OTP):
    """GroupNorm via Reshape(N,G,C/G,...) → LN → Reshape. TVM reshape+norm bugs."""
    name = "manual_group_norm"
    category = CAT_NORMALIZATION
    target_optimization = "group_norm_lowering"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        C = ctx.shape[1]
        return C >= 4 and C % 4 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        G = 4   # number of groups
        p = self._p(node_id, "mgn")

        # Reshape to [N*G, C//G*H*W] for LN, then back
        group_size = (C // G) * H * W
        s_group = np.array([N * G, group_size], dtype=np.int64)
        s_back  = np.array([N, C, H, W],        dtype=np.int64)
        ln_sc   = np.ones(group_size,  dtype=np.float32)
        ln_b    = np.zeros(group_size, dtype=np.float32)
        out_sc  = np.ones((1, C, 1, 1), dtype=np.float32)
        out_b   = np.zeros((1, C, 1, 1), dtype=np.float32)

        rg_o = f"{p}_rg"; ln_o = f"{p}_ln"; rb_o = f"{p}_rb"
        sc_o = f"{p}_sc"; out  = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_sg"], [rg_o]),
            helper.make_node("LayerNormalization",
                             [rg_o, f"{p}_lnsc", f"{p}_lnb"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Reshape", [ln_o, f"{p}_sb"], [rb_o]),
            helper.make_node("Mul",     [rb_o, f"{p}_osc"], [sc_o]),
            helper.make_node("Add",     [sc_o, f"{p}_ob"],  [out]),
        ]
        inits = [numpy_helper.from_array(s_group, f"{p}_sg"),
                 numpy_helper.from_array(s_back,  f"{p}_sb"),
                 numpy_helper.from_array(ln_sc,   f"{p}_lnsc"),
                 numpy_helper.from_array(ln_b,    f"{p}_lnb"),
                 numpy_helper.from_array(out_sc,  f"{p}_osc"),
                 numpy_helper.from_array(out_b,   f"{p}_ob")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 12. L2 Normalize ──────────────────────────────────────────────────────
class L2Normalize(OTP):
    """L2 normalization: x / ||x||_2. Tests ReduceL2 + Div fusion."""
    name = "l2_normalize"
    category = CAT_NORMALIZATION
    target_optimization = "l2_norm_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "l2n2")
        eps = np.array([1e-12], dtype=np.float32)
        scale = np.ones([1] * ctx.rank, dtype=np.float32)

        norm_o = f"{p}_norm"; add_o = f"{p}_add"; div_o = f"{p}_div"; out = f"{p}_out"
        nodes = [
            helper.make_node("ReduceL2",  [input_name], [norm_o],
                             axes=[-1], keepdims=1),
            helper.make_node("Add",       [norm_o, f"{p}_eps"], [add_o]),
            helper.make_node("Div",       [input_name, add_o], [div_o]),
            helper.make_node("Mul",       [div_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(eps,   f"{p}_eps"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 13. Power Normalization (RMS-style, non-standard) ─────────────────────
class PowerNorm(OTP):
    """x / (mean(|x|^p)^(1/p) + eps): power normalization variant."""
    name = "power_norm"
    category = CAT_NORMALIZATION
    target_optimization = "power_norm_lowering"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "pnm")
        half = np.array([0.5],  dtype=np.float32)
        eps  = np.array([1e-6], dtype=np.float32)
        scale = np.ones(ctx.shape, dtype=np.float32)

        sq_o = f"{p}_sq"; mean_o = f"{p}_mean"; add_o = f"{p}_add"
        pow_o = f"{p}_pow"; div_o = f"{p}_div"; out = f"{p}_out"
        nodes = [
            # Mul(x,x) avoids Pow-on-negative-base UB (would NaN on some backends).
            helper.make_node("Mul",       [input_name, input_name], [sq_o]),
            helper.make_node("ReduceMean",[sq_o], [mean_o], axes=[-1], keepdims=1),
            helper.make_node("Add",       [mean_o, f"{p}_eps"], [add_o]),
            # add_o is mean(x^2)+eps > 0, so Pow(.,0.5) is well-defined here.
            helper.make_node("Pow",       [add_o, f"{p}_half"], [pow_o]),
            helper.make_node("Div",       [input_name, pow_o], [div_o]),
            helper.make_node("Mul",       [div_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(half,  f"{p}_half"),
                 numpy_helper.from_array(eps,   f"{p}_eps"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 14. AdaLayerNorm (scale/shift from constants) ─────────────────────────
class AdaLayerNorm(OTP):
    """Adaptive LayerNorm: scale and bias from a separate affine map."""
    name = "ada_layer_norm"
    category = CAT_NORMALIZATION
    target_optimization = "ada_norm_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "aln")
        D = ctx.shape[-1]
        # LN first
        ln_sc = np.ones(D,  dtype=np.float32)
        ln_b  = np.zeros(D, dtype=np.float32)
        # Adaptive modulation constants (in real use, from a condition signal)
        ada_sc = np.ones([1] * (ctx.rank - 1) + [D], dtype=np.float32)
        ada_b  = np.zeros([1] * (ctx.rank - 1) + [D], dtype=np.float32)

        ln_o = f"{p}_ln"; sc_o = f"{p}_sc"; out = f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_lnsc", f"{p}_lnb"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Mul", [ln_o, f"{p}_asc"], [sc_o]),
            helper.make_node("Add", [sc_o, f"{p}_ab"], [out]),
        ]
        inits = [numpy_helper.from_array(ln_sc,  f"{p}_lnsc"),
                 numpy_helper.from_array(ln_b,   f"{p}_lnb"),
                 numpy_helper.from_array(ada_sc, f"{p}_asc"),
                 numpy_helper.from_array(ada_b,  f"{p}_ab")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 15. BatchNorm + Relu6 → Fold + Clip ──────────────────────────────────
class BatchNormReLU6(OTP):
    """BN → ReLU6: triggers BN-folding + Clip pass interaction bugs."""
    name = "batchnorm_relu6"
    category = CAT_NORMALIZATION
    target_optimization = "bn_relu6_fold"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "bnr6")
        scale = np.ones(C, dtype=np.float32)
        bias  = np.zeros(C, dtype=np.float32)
        mean  = np.zeros(C, dtype=np.float32)
        var   = np.ones(C, dtype=np.float32)
        zero  = np.array([0.0], dtype=np.float32)
        six   = np.array([6.0], dtype=np.float32)

        bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("BatchNormalization",
                             [input_name, f"{p}_sc", f"{p}_b", f"{p}_m", f"{p}_v"],
                             [bn_o], epsilon=1e-5),
            helper.make_node("Clip", [bn_o, f"{p}_zero", f"{p}_six"], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b"),
                 numpy_helper.from_array(mean,  f"{p}_m"),
                 numpy_helper.from_array(var,   f"{p}_v"),
                 numpy_helper.from_array(zero,  f"{p}_zero"),
                 numpy_helper.from_array(six,   f"{p}_six")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 16. Variance-based Whitening ─────────────────────────────────────────
class VarianceWhitening(OTP):
    """(x - mean) / std: manual whitening without LN op. Arithmetic chain bugs."""
    name = "variance_whitening"
    category = CAT_NORMALIZATION
    target_optimization = "whitening_normalization_fold"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "vw")
        eps = np.array([1e-5], dtype=np.float32)
        half = np.array([0.5], dtype=np.float32)
        sc = np.ones([1]*ctx.rank, dtype=np.float32)

        mean_o = f"{p}_mean"; diff_o = f"{p}_diff"
        sq_o   = f"{p}_sq";   var_o  = f"{p}_var"
        add_o  = f"{p}_add";  std_o  = f"{p}_std"; div_o = f"{p}_div"; out = f"{p}_out"
        nodes = [
            helper.make_node("ReduceMean", [input_name], [mean_o], axes=[-1], keepdims=1),
            helper.make_node("Sub",        [input_name, mean_o], [diff_o]),
            helper.make_node("Mul",        [diff_o, diff_o], [sq_o]),
            helper.make_node("ReduceMean", [sq_o], [var_o], axes=[-1], keepdims=1),
            helper.make_node("Add",        [var_o, f"{p}_eps"], [add_o]),
            helper.make_node("Pow",        [add_o, f"{p}_half"], [std_o]),
            helper.make_node("Div",        [diff_o, std_o], [div_o]),
            helper.make_node("Mul",        [div_o, f"{p}_sc"], [out]),
        ]
        inits = [numpy_helper.from_array(eps,  f"{p}_eps"),
                 numpy_helper.from_array(half, f"{p}_half"),
                 numpy_helper.from_array(sc,   f"{p}_sc")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 17. InstanceNorm on 1D conv output [N, C, L] ─────────────────────────
class InstanceNorm1D(OTP):
    """InstanceNorm applied to [N, C, L] (1-D conv output). Shape mismatch bugs."""
    name = "instance_norm_1d"
    category = CAT_NORMALIZATION
    target_optimization = "instance_norm_1d_lowering"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, M = ctx.shape
        L = M   # treat as sequence
        C = 1
        p = self._p(node_id, "in1d")

        # Reshape [N, M] → [N, 1, L] for InstanceNorm, then back
        s3d = np.array([N, C, L], dtype=np.int64)
        s2d = np.array([N, M],    dtype=np.int64)
        sc  = np.ones(C,  dtype=np.float32)
        b   = np.zeros(C, dtype=np.float32)

        r3_o = f"{p}_r3"; in_o = f"{p}_in"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape",      [input_name, f"{p}_s3d"], [r3_o]),
            helper.make_node("InstanceNormalization",
                             [r3_o, f"{p}_sc", f"{p}_b"], [in_o],
                             epsilon=1e-5),
            helper.make_node("Reshape",      [in_o, f"{p}_s2d"], [out]),
        ]
        inits = [numpy_helper.from_array(s3d, f"{p}_s3d"),
                 numpy_helper.from_array(s2d, f"{p}_s2d"),
                 numpy_helper.from_array(sc,  f"{p}_sc"),
                 numpy_helper.from_array(b,   f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, M], ctx.dtype, ctx.layout))


# ── 18. LayerNorm + Learnable Temperature ────────────────────────────────
class LayerNormTemperature(OTP):
    """LayerNorm followed by learned temperature scaling (Div by constant)."""
    name = "layernorm_temperature"
    category = CAT_NORMALIZATION
    target_optimization = "layernorm_scale_fold"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "lnt")
        D = ctx.shape[-1]
        sc  = np.ones(D,  dtype=np.float32)
        b   = np.zeros(D, dtype=np.float32)
        temp = np.array([2.25], dtype=np.float32)
        bias = np.zeros(ctx.shape, dtype=np.float32)

        ln_o = f"{p}_ln"; div_o = f"{p}_div"; out = f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_sc", f"{p}_b"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Div", [ln_o, f"{p}_temp"], [div_o]),
            helper.make_node("Add", [div_o, f"{p}_bias"], [out]),
        ]
        inits = [numpy_helper.from_array(sc,   f"{p}_sc"),
                 numpy_helper.from_array(b,    f"{p}_b"),
                 numpy_helper.from_array(temp, f"{p}_temp"),
                 numpy_helper.from_array(bias, f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 19. ScaledRMSNorm with learnable gamma + beta ────────────────────────
class ScaledRMSNorm(OTP):
    """RMSNorm(x) * gamma + beta: full affine variant used in LLaMA/Mistral."""
    name = "scaled_rmsnorm"
    category = CAT_NORMALIZATION
    target_optimization = "rmsnorm_affine_lowering"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "srms")
        D   = ctx.shape[-1]
        eps = np.array([1e-6], dtype=np.float32)
        g   = np.ones(D,  dtype=np.float32)
        b   = np.zeros(D, dtype=np.float32)

        sq_o = f"{p}_sq"; mn_o = f"{p}_mn"; add_o = f"{p}_add"
        sq2_o = f"{p}_sq2"; rc_o = f"{p}_rc"; nm_o = f"{p}_nm"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul",        [input_name, input_name], [sq_o]),
            helper.make_node("ReduceMean", [sq_o], [mn_o], axes=[-1], keepdims=1),
            helper.make_node("Add",        [mn_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [sq2_o]),
            helper.make_node("Reciprocal", [sq2_o], [rc_o]),
            helper.make_node("Mul",        [input_name, rc_o], [nm_o]),
            helper.make_node("Mul",        [nm_o, f"{p}_g"], [f"{p}_gm"]),
            helper.make_node("Add",        [f"{p}_gm", f"{p}_b"], [out]),
        ]
        inits = [numpy_helper.from_array(eps, f"{p}_eps"),
                 numpy_helper.from_array(g,   f"{p}_g"),
                 numpy_helper.from_array(b,   f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 20. Pre-norm FFN block: LN → Linear → GELU → Linear ─────────────────
class PreNormFFN(OTP):
    """LayerNorm + Linear + GELU + Linear: standard transformer FFN block."""
    name = "prenorm_ffn"
    category = CAT_NORMALIZATION
    target_optimization = "prenorm_ffn_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        ff_dim = D * 4
        p = self._p(node_id, "pnffn")
        g  = np.ones(D, dtype=np.float32)
        b  = np.zeros(D, dtype=np.float32)
        w1 = self._make_linear_weight(D, ff_dim)
        b1 = np.zeros(ff_dim, dtype=np.float32)
        w2 = self._make_linear_weight(ff_dim, D)
        b2 = np.zeros(D, dtype=np.float32)
        sqrt2 = np.array([float(np.sqrt(2.0))], dtype=np.float32)
        half  = np.array([0.5], dtype=np.float32)
        one   = np.array([1.0], dtype=np.float32)

        ln_o  = f"{p}_ln";  mm1_o = f"{p}_mm1"; a1_o = f"{p}_a1"
        gd_o  = f"{p}_gd";  erf_o = f"{p}_erf"; ep1_o = f"{p}_ep1"
        hf_o  = f"{p}_hf";  ge_o  = f"{p}_ge";  mm2_o = f"{p}_mm2"
        out   = f"{p}_out"

        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_g", f"{p}_b"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("MatMul", [ln_o, f"{p}_w1"], [mm1_o]),
            helper.make_node("Add",    [mm1_o, f"{p}_b1"], [a1_o]),
            # GELU activation
            helper.make_node("Div",    [a1_o, f"{p}_sqrt2"], [gd_o]),
            helper.make_node("Erf",    [gd_o], [erf_o]),
            helper.make_node("Add",    [erf_o, f"{p}_one"], [ep1_o]),
            helper.make_node("Mul",    [a1_o, f"{p}_half"], [hf_o]),
            helper.make_node("Mul",    [hf_o, ep1_o], [ge_o]),
            helper.make_node("MatMul", [ge_o, f"{p}_w2"], [mm2_o]),
            helper.make_node("Add",    [mm2_o, f"{p}_b2"], [out]),
        ]
        inits = [numpy_helper.from_array(g,     f"{p}_g"),
                 numpy_helper.from_array(b,     f"{p}_b"),
                 numpy_helper.from_array(w1,    f"{p}_w1"),
                 numpy_helper.from_array(b1,    f"{p}_b1"),
                 numpy_helper.from_array(w2,    f"{p}_w2"),
                 numpy_helper.from_array(b2,    f"{p}_b2"),
                 numpy_helper.from_array(sqrt2, f"{p}_sqrt2"),
                 numpy_helper.from_array(half,  f"{p}_half"),
                 numpy_helper.from_array(one,   f"{p}_one")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, ctx.layout))


# ── 21. Double LayerNorm (sandwiched normalization) ───────────────────────
class DoubleLayerNorm(OTP):
    """LN → transform → LN: used in some ViT/Swin variants. Tests LN-LN fusion."""
    name = "double_layernorm"
    category = CAT_NORMALIZATION
    target_optimization = "double_layernorm_fusion"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "dln")
        D = ctx.shape[-1]
        g1 = np.ones(D,  dtype=np.float32)
        b1 = np.zeros(D, dtype=np.float32)
        g2 = np.ones(D,  dtype=np.float32)
        b2 = np.zeros(D, dtype=np.float32)
        scale = np.full(list(ctx.shape), 1.1, dtype=np.float32)

        ln1_o = f"{p}_ln1"; sc_o = f"{p}_sc"; out = f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_g1", f"{p}_b1"], [ln1_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Mul", [ln1_o, f"{p}_scale"], [sc_o]),
            helper.make_node("LayerNormalization",
                             [sc_o, f"{p}_g2", f"{p}_b2"], [out],
                             axis=-1, epsilon=1e-5),
        ]
        inits = [numpy_helper.from_array(g1,    f"{p}_g1"),
                 numpy_helper.from_array(b1,    f"{p}_b1"),
                 numpy_helper.from_array(g2,    f"{p}_g2"),
                 numpy_helper.from_array(b2,    f"{p}_b2"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 22. Conditional RMSNorm (scale from context) ──────────────────────────
class CRMSNorm(OTP):
    """RMSNorm + element-wise multiply by learned scale: conditional norm."""
    name = "crms_norm"
    category = CAT_NORMALIZATION
    target_optimization = "conditional_rmsnorm_fusion"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        D = ctx.shape[-1]
        p = self._p(node_id, "crms")
        g  = np.ones(D,  dtype=np.float32)
        g2 = (np.random.default_rng(node_id).standard_normal(D) * 0.1 + 1.0).astype(np.float32)
        eps = np.array([1e-6], dtype=np.float32)

        sq_o=f"{p}_sq"; ms_o=f"{p}_ms"; add_o=f"{p}_add"
        rt_o=f"{p}_rt"; n_o=f"{p}_n"; sc_o=f"{p}_sc"; out=f"{p}_out"
        # opset 17 keeps ReduceMean `axes` as an ATTRIBUTE (axes-as-input
        # was added only in opset 18).
        nodes = [
            helper.make_node("Mul",       [input_name, input_name], [sq_o]),
            helper.make_node("ReduceMean",[sq_o], [ms_o], axes=[-1], keepdims=1),
            helper.make_node("Add",       [ms_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",      [add_o], [rt_o]),
            helper.make_node("Div",       [input_name, rt_o], [n_o]),
            helper.make_node("Mul",       [n_o, f"{p}_g"], [sc_o]),
            helper.make_node("Mul",       [sc_o, f"{p}_g2"], [out]),
        ]
        inits = [numpy_helper.from_array(g,    f"{p}_g"),
                 numpy_helper.from_array(g2,   f"{p}_g2"),
                 numpy_helper.from_array(eps,  f"{p}_eps")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout))


# ── 23. LayerNorm → Gate ────────────────────────────────────────────────────
class LayerNormGate(OTP):
    """LayerNorm → Sigmoid gate → Mul: gated layer norm pattern."""
    name = "layernorm_gate"
    category = CAT_NORMALIZATION
    target_optimization = "layernorm_gate_fusion"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        D = ctx.shape[-1]
        p = self._p(node_id, "lng")
        g = np.ones(D, dtype=np.float32)
        b = np.zeros(D, dtype=np.float32)

        ln_o=f"{p}_ln"; sig_o=f"{p}_sig"; out=f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_g", f"{p}_b"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Sigmoid", [ln_o], [sig_o]),
            helper.make_node("Mul",     [ln_o, sig_o], [out]),
        ]
        inits = [numpy_helper.from_array(g, f"{p}_g"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout))


# ── 24. GroupNorm + SiLU ────────────────────────────────────────────────────
class GroupNormSiLU(OTP):
    """Manual group norm + SiLU: stable diffusion U-Net pattern."""
    name = "groupnorm_silu"
    category = CAT_NORMALIZATION
    target_optimization = "groupnorm_silu_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        C = ctx.shape[1]
        return C >= 8 and C % 8 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        G = min(8, C)
        p = self._p(node_id, "gns")
        # Channel affine broadcast to [N,C,H,W] requires shape (1,C,1,1).
        g = np.ones((1, C, 1, 1), dtype=np.float32)
        b = np.zeros((1, C, 1, 1), dtype=np.float32)
        rs1 = np.array([N, G, C//G, H, W],  dtype=np.int64)
        rs2 = np.array([N, C, H, W],         dtype=np.int64)
        eps  = np.array([1e-5], dtype=np.float32)

        r1_o=f"{p}_r1"; mu_o=f"{p}_mu"; sq_o=f"{p}_sq"
        ms_o=f"{p}_ms"; add_o=f"{p}_add"; rt_o=f"{p}_rt"
        sub_o=f"{p}_sub"; div_o=f"{p}_div"; r2_o=f"{p}_r2"
        sc_o=f"{p}_sc"; sig_o=f"{p}_sig"; out=f"{p}_out"
        # opset 17 ReduceMean takes `axes` as an ATTRIBUTE, not an input.
        nodes = [
            helper.make_node("Reshape",    [input_name, f"{p}_rs1"], [r1_o]),
            helper.make_node("ReduceMean", [r1_o], [mu_o], axes=[2, 3, 4], keepdims=1),
            helper.make_node("Sub",        [r1_o, mu_o], [sub_o]),
            helper.make_node("Mul",        [sub_o, sub_o], [sq_o]),
            helper.make_node("ReduceMean", [sq_o], [ms_o], axes=[2, 3, 4], keepdims=1),
            helper.make_node("Add",        [ms_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [rt_o]),
            helper.make_node("Div",        [sub_o, rt_o], [div_o]),
            helper.make_node("Reshape",    [div_o, f"{p}_rs2"], [r2_o]),
            helper.make_node("Mul",        [r2_o, f"{p}_g"], [sc_o]),
            helper.make_node("Add",        [sc_o, f"{p}_b"], [f"{p}_gn"]),
            # SiLU: x * sigmoid(x)
            helper.make_node("Sigmoid",    [f"{p}_gn"], [sig_o]),
            helper.make_node("Mul",        [f"{p}_gn", sig_o], [out]),
        ]
        inits = [numpy_helper.from_array(g,    f"{p}_g"),
                 numpy_helper.from_array(b,    f"{p}_b"),
                 numpy_helper.from_array(rs1,  f"{p}_rs1"),
                 numpy_helper.from_array(rs2,  f"{p}_rs2"),
                 numpy_helper.from_array(eps,  f"{p}_eps")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


ALL_NORMALIZATION_PATTERNS = [
    ManualLayerNorm(),
    LayerNormReLU(),
    LayerNormResidualAdd(),
    InstanceNormReLU(),
    RMSNorm(),
    GroupNormReLU(),
    BatchNormEval(),
    StableSoftmax(),
    SpatialReduceMean(),
    LayerNormDropout(),
    ManualGroupNorm(),
    L2Normalize(),
    PowerNorm(),
    AdaLayerNorm(),
    BatchNormReLU6(),
    VarianceWhitening(),
    InstanceNorm1D(),
    LayerNormTemperature(),
    ScaledRMSNorm(),
    PreNormFFN(),
    DoubleLayerNorm(),
    CRMSNorm(),
    LayerNormGate(),
    GroupNormSiLU(),
]
