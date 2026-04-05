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
        res   = rng.normal(0, 0.1, ctx.shape).astype(np.float32)

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
        two = np.array([2.0],  dtype=np.float32)
        axes = [-1]

        sq_o = f"{p}_sq"; mean_o = f"{p}_mean"
        add_o = f"{p}_add"; sqrt_o = f"{p}_sqrt"
        sc_o = f"{p}_sc"; out = f"{p}_out"

        norm_dim = ctx.shape[-1]
        g = np.ones(norm_dim, dtype=np.float32)

        nodes = [
            helper.make_node("Pow",        [input_name, f"{p}_two"], [sq_o]),
            helper.make_node("ReduceMean", [sq_o], [mean_o],
                             axes=axes, keepdims=1),
            helper.make_node("Add",        [mean_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [sqrt_o]),
            helper.make_node("Div",        [input_name, sqrt_o], [sc_o]),
            helper.make_node("Mul",        [sc_o, f"{p}_g"], [out]),
        ]
        inits = [numpy_helper.from_array(eps, f"{p}_eps"),
                 numpy_helper.from_array(two, f"{p}_two"),
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
        scale = rng.normal(1, 0.1, C).astype(np.float32)
        bias  = rng.normal(0, 0.1, C).astype(np.float32)
        mean  = rng.normal(0, 0.5, C).astype(np.float32)
        var   = np.abs(rng.normal(1, 0.2, C)).astype(np.float32) + 1e-5

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
        axes = [-1]

        mx_o = f"{p}_mx"; sub_o = f"{p}_sub"; exp_o = f"{p}_exp"
        sum_o = f"{p}_sum"; out = f"{p}_out"
        nodes = [
            helper.make_node("ReduceMax", [input_name], [mx_o],
                             axes=axes, keepdims=1),
            helper.make_node("Sub",       [input_name, mx_o], [sub_o]),
            helper.make_node("Exp",       [sub_o], [exp_o]),
            helper.make_node("ReduceSum", [exp_o], [sum_o],
                             axes=axes, keepdims=1),
            helper.make_node("Div",       [exp_o, sum_o], [out]),
        ]
        return PatternInstance(self.name, self.category, nodes, [],
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
        scale = rng.normal(1, 0.1, (1, C, 1, 1)).astype(np.float32)

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
        ratio = np.array([0.0], dtype=np.float32)  # dropout off in inference

        ln_o = f"{p}_ln"; out = f"{p}_out"
        nodes = [
            helper.make_node("LayerNormalization",
                             [input_name, f"{p}_sc", f"{p}_b"], [ln_o],
                             axis=-1, epsilon=1e-5),
            # Inference dropout (ratio=0) should be eliminated by compilers
            helper.make_node("Dropout", [ln_o, f"{p}_ratio"], [out],
                             seed=0),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b"),
                 numpy_helper.from_array(ratio, f"{p}_ratio")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


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
]
