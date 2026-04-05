"""
Branch- and aggregation-sensitive OTPs.
Target: branch merging, dependency rewrites, aggregation correctness.
"""
from __future__ import annotations
import numpy as np
from onnx import helper, numpy_helper
from typing import Optional

from .base import OTP, PatternInstance, CAT_BRANCH
from ..generation.context import StructuralContext


# ── 1. Split → Transform → Concat ─────────────────────────────────────────
class SplitTransformConcat(OTP):
    name = "split_transform_concat"
    category = CAT_BRANCH
    target_optimization = "split_concat_elimination"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32" and ctx.channels() % 2 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        half = C // 2
        p = self._p(node_id, "stc")

        split_sizes = np.array([half, half], dtype=np.int64)
        w1 = np.ones((half, 1, 1, 1), dtype=np.float32) * float(rng.uniform(0.8, 1.2))
        w2 = np.ones((half, 1, 1, 1), dtype=np.float32) * float(rng.uniform(0.8, 1.2))

        a_o = f"{p}_a"; b_o = f"{p}_b"
        ma_o = f"{p}_ma"; mb_o = f"{p}_mb"; out = f"{p}_out"
        nodes = [
            helper.make_node("Split", [input_name, f"{p}_sp"],
                             [a_o, b_o], axis=1),
            helper.make_node("Mul",  [a_o, f"{p}_w1"], [ma_o]),
            helper.make_node("Mul",  [b_o, f"{p}_w2"], [mb_o]),
            helper.make_node("Concat", [ma_o, mb_o], [out], axis=1),
        ]
        inits = [numpy_helper.from_array(split_sizes, f"{p}_sp"),
                 numpy_helper.from_array(w1, f"{p}_w1"),
                 numpy_helper.from_array(w2, f"{p}_w2")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 2. Residual Add → ReLU (ResNet block end) ─────────────────────────────
class ResidualAddReLU(OTP):
    name = "residual_add_relu"
    category = CAT_BRANCH
    target_optimization = "residual_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32" and ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "rarelu")
        residual = rng.normal(0, 0.1, ctx.shape).astype(np.float32)

        add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Add",  [input_name, f"{p}_res"], [add_o]),
            helper.make_node("Relu", [add_o], [out]),
        ]
        inits = [numpy_helper.from_array(residual, f"{p}_res")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 3. Conv → Branch → Add (residual shortcut with 1×1 projection) ────────
class ConvBranchAdd(OTP):
    name = "conv_branch_add"
    category = CAT_BRANCH
    target_optimization = "branch_add_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "cba")

        # Main branch: 3×3 conv
        w_main = self._make_conv_weight(rng, out_c, C, 3)
        b_main = np.zeros(out_c, dtype=np.float32)
        # Skip branch: 1×1 projection
        w_skip = self._make_conv_weight(rng, out_c, C, 1)
        b_skip = np.zeros(out_c, dtype=np.float32)

        main_o = f"{p}_main"; skip_o = f"{p}_skip"
        relu_m = f"{p}_relum"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_wm", f"{p}_bm"], [main_o],
                             kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]),
            helper.make_node("Relu", [main_o], [relu_m]),
            helper.make_node("Conv", [input_name, f"{p}_ws", f"{p}_bs"], [skip_o],
                             kernel_shape=[1,1], pads=[0,0,0,0], strides=[1,1]),
            helper.make_node("Add", [relu_m, skip_o], [out]),
        ]
        inits = [numpy_helper.from_array(w_main, f"{p}_wm"),
                 numpy_helper.from_array(b_main, f"{p}_bm"),
                 numpy_helper.from_array(w_skip, f"{p}_ws"),
                 numpy_helper.from_array(b_skip, f"{p}_bs")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H,W], ctx.dtype, ctx.layout))


# ── 4. MaxPool + AvgPool → Add (dual-path pooling) ────────────────────────
class DualPoolAdd(OTP):
    name = "dual_pool_add"
    category = CAT_BRANCH
    target_optimization = "pool_aggregation"

    def is_compatible(self, ctx):
        return (ctx.rank == 4 and ctx.dtype == "float32"
                and ctx.shape[2] >= 3 and ctx.shape[3] >= 3)

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "dpa")

        maxp_o = f"{p}_maxp"; avgp_o = f"{p}_avgp"; out = f"{p}_out"
        nodes = [
            helper.make_node("MaxPool",     [input_name], [maxp_o],
                             kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1]),
            helper.make_node("AveragePool", [input_name], [avgp_o],
                             kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1]),
            helper.make_node("Add", [maxp_o, avgp_o], [out]),
        ]
        return PatternInstance(self.name, self.category, nodes, [],
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 5. SE block: GlobalAvgPool → Linear → Sigmoid → Mul ──────────────────
class SEBlock(OTP):
    name = "se_block"
    category = CAT_BRANCH
    target_optimization = "se_block_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        r = max(C // 4, 1)   # reduction ratio
        p = self._p(node_id, "seb")

        w1 = rng.normal(0, np.sqrt(2/C), (C, r)).astype(np.float32)
        b1 = np.zeros(r, dtype=np.float32)
        w2 = rng.normal(0, np.sqrt(2/r), (r, C)).astype(np.float32)
        b2 = np.zeros(C, dtype=np.float32)
        rs = np.array([-1, C, 1, 1], dtype=np.int64)

        gap_o = f"{p}_gap"; fl_o = f"{p}_fl"
        mm1_o = f"{p}_mm1"; relu_o = f"{p}_relu"
        mm2_o = f"{p}_mm2"; sig_o = f"{p}_sig"
        att_o = f"{p}_att"; out = f"{p}_out"
        nodes = [
            helper.make_node("GlobalAveragePool", [input_name], [gap_o]),
            helper.make_node("Flatten",  [gap_o], [fl_o], axis=1),
            helper.make_node("MatMul",   [fl_o, f"{p}_w1"], [mm1_o]),
            helper.make_node("Add",      [mm1_o, f"{p}_b1"], [f"{p}_add1"]),
            helper.make_node("Relu",     [f"{p}_add1"], [relu_o]),
            helper.make_node("MatMul",   [relu_o, f"{p}_w2"], [mm2_o]),
            helper.make_node("Add",      [mm2_o, f"{p}_b2"], [f"{p}_add2"]),
            helper.make_node("Sigmoid",  [f"{p}_add2"], [sig_o]),
            helper.make_node("Reshape",  [sig_o, f"{p}_rs"], [att_o]),
            helper.make_node("Mul",      [input_name, att_o], [out]),
        ]
        inits = [numpy_helper.from_array(w1, f"{p}_w1"),
                 numpy_helper.from_array(b1, f"{p}_b1"),
                 numpy_helper.from_array(w2, f"{p}_w2"),
                 numpy_helper.from_array(b2, f"{p}_b2"),
                 numpy_helper.from_array(rs, f"{p}_rs")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 6. Split → Activation → Concat (gated activation, e.g. GLU) ──────────
class GatedLinearUnit(OTP):
    name = "glu"
    category = CAT_BRANCH
    target_optimization = "gated_activation_fusion"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32" and ctx.shape[-1] % 2 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "glu")
        half = ctx.shape[-1] // 2
        out_shape = list(ctx.shape[:-1]) + [half]

        split_sizes = np.array([half, half], dtype=np.int64)

        a_o = f"{p}_a"; b_o = f"{p}_b"; sig_o = f"{p}_sig"; out = f"{p}_out"
        nodes = [
            helper.make_node("Split", [input_name, f"{p}_sp"],
                             [a_o, b_o], axis=-1),
            helper.make_node("Sigmoid", [b_o], [sig_o]),
            helper.make_node("Mul",     [a_o, sig_o], [out]),
        ]
        inits = [numpy_helper.from_array(split_sizes, f"{p}_sp")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(len(out_shape), out_shape,
                                                 ctx.dtype, ctx.layout))


# ── 7. Add → LayerNorm (post-norm residual) ───────────────────────────────
class AddLayerNorm(OTP):
    name = "add_layernorm"
    category = CAT_BRANCH
    target_optimization = "post_norm_residual"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "aln")
        res = rng.normal(0, 0.1, ctx.shape).astype(np.float32)
        norm_dim = ctx.shape[-1]
        scale = np.ones(norm_dim,  dtype=np.float32)
        bias  = np.zeros(norm_dim, dtype=np.float32)

        add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Add", [input_name, f"{p}_res"], [add_o]),
            helper.make_node("LayerNormalization",
                             [add_o, f"{p}_sc", f"{p}_b"], [out],
                             axis=-1, epsilon=1e-5),
        ]
        inits = [numpy_helper.from_array(res,   f"{p}_res"),
                 numpy_helper.from_array(scale, f"{p}_sc"),
                 numpy_helper.from_array(bias,  f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 8. Concat → Conv (feature aggregation then fused conv) ────────────────
class ConcatConv(OTP):
    name = "concat_conv"
    category = CAT_BRANCH
    target_optimization = "concat_conv_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        extra_c = self._rand_channels(rng)
        out_c   = self._rand_channels(rng)
        p = self._p(node_id, "ccv")

        extra = rng.normal(0, 0.1, (N, extra_c, H, W)).astype(np.float32)
        w = self._make_conv_weight(rng, out_c, C + extra_c, 1)
        b = np.zeros(out_c, dtype=np.float32)

        cat_o = f"{p}_cat"; out = f"{p}_out"
        nodes = [
            helper.make_node("Concat", [input_name, f"{p}_extra"], [cat_o], axis=1),
            helper.make_node("Conv",   [cat_o, f"{p}_w", f"{p}_b"], [out],
                             kernel_shape=[1,1], pads=[0,0,0,0], strides=[1,1]),
        ]
        inits = [numpy_helper.from_array(extra, f"{p}_extra"),
                 numpy_helper.from_array(w,     f"{p}_w"),
                 numpy_helper.from_array(b,     f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H,W], ctx.dtype, ctx.layout))


ALL_BRANCH_PATTERNS = [
    SplitTransformConcat(),
    ResidualAddReLU(),
    ConvBranchAdd(),
    DualPoolAdd(),
    SEBlock(),
    GatedLinearUnit(),
    AddLayerNorm(),
    ConcatConv(),
]
