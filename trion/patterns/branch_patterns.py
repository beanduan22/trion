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
        w1 = np.ones((1, half, 1, 1), dtype=np.float32) * float(rng.uniform(0.8, 1.2))
        w2 = np.ones((1, half, 1, 1), dtype=np.float32) * float(rng.uniform(0.8, 1.2))

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


# ── 9. Multi-Scale Conv Branches (Inception-style) ───────────────────────
class MultiScaleConvBranch(OTP):
    """3 parallel convs (1×1, 3×3, 5×5) + element-wise add."""
    name = "multi_scale_conv_branch"
    category = CAT_BRANCH
    target_optimization = "parallel_conv_merging"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "mscb")

        w1 = self._make_conv_weight(rng, out_c, C, 1)
        w3 = self._make_conv_weight(rng, out_c, C, 3)
        w5 = self._make_conv_weight(rng, out_c, C, 5)
        b1 = np.zeros(out_c, dtype=np.float32)
        b3 = np.zeros(out_c, dtype=np.float32)
        b5 = np.zeros(out_c, dtype=np.float32)

        br1 = f"{p}_br1"; br3 = f"{p}_br3"; br5 = f"{p}_br5"
        add1 = f"{p}_a1"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w1", f"{p}_b1"], [br1],
                             kernel_shape=[1,1], pads=[0]*4, strides=[1,1]),
            helper.make_node("Conv", [input_name, f"{p}_w3", f"{p}_b3"], [br3],
                             kernel_shape=[3,3], pads=[1]*4, strides=[1,1]),
            helper.make_node("Conv", [input_name, f"{p}_w5", f"{p}_b5"], [br5],
                             kernel_shape=[5,5], pads=[2]*4, strides=[1,1]),
            helper.make_node("Add",  [br1, br3], [add1]),
            helper.make_node("Add",  [add1, br5], [out]),
        ]
        inits = [numpy_helper.from_array(w1, f"{p}_w1"),
                 numpy_helper.from_array(b1, f"{p}_b1"),
                 numpy_helper.from_array(w3, f"{p}_w3"),
                 numpy_helper.from_array(b3, f"{p}_b3"),
                 numpy_helper.from_array(w5, f"{p}_w5"),
                 numpy_helper.from_array(b5, f"{p}_b5")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H, W], ctx.dtype, ctx.layout))


# ── 10. ASPP-style Dilated Branch ─────────────────────────────────────────
class ASPPDilatedBranch(OTP):
    """3 dilated convs (d=1,2,4) + concat + 1×1. DeepLab ASPP pattern."""
    name = "aspp_dilated_branch"
    category = CAT_BRANCH
    target_optimization = "dilated_branch_merging"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H >= 8 and W >= 8

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        br_c = max(C // 3, 8)
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "aspp")

        w1 = self._make_conv_weight(rng, br_c, C, 1)
        w2 = self._make_conv_weight(rng, br_c, C, 3)
        w3 = self._make_conv_weight(rng, br_c, C, 3)
        wm = self._make_conv_weight(rng, out_c, br_c * 3, 1)
        for w, b_n in [(w1, "b1"), (w2, "b2"), (w3, "b3"), (wm, "bm")]:
            pass  # just need the shapes

        br1 = f"{p}_br1"; br2 = f"{p}_br2"; br3 = f"{p}_br3"
        cat = f"{p}_cat"; out = f"{p}_out"
        b1 = np.zeros(br_c,  dtype=np.float32)
        b2 = np.zeros(br_c,  dtype=np.float32)
        b3 = np.zeros(br_c,  dtype=np.float32)
        bm = np.zeros(out_c, dtype=np.float32)
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w1", f"{p}_b1"], [br1],
                             kernel_shape=[1,1], pads=[0]*4, strides=[1,1]),
            helper.make_node("Conv", [input_name, f"{p}_w2", f"{p}_b2"], [br2],
                             kernel_shape=[3,3], pads=[2]*4, strides=[1,1],
                             dilations=[2,2]),
            helper.make_node("Conv", [input_name, f"{p}_w3", f"{p}_b3"], [br3],
                             kernel_shape=[3,3], pads=[4]*4, strides=[1,1],
                             dilations=[4,4]),
            helper.make_node("Concat", [br1, br2, br3], [cat], axis=1),
            helper.make_node("Conv",   [cat, f"{p}_wm", f"{p}_bm"], [out],
                             kernel_shape=[1,1], pads=[0]*4, strides=[1,1]),
        ]
        inits = [numpy_helper.from_array(w1, f"{p}_w1"),
                 numpy_helper.from_array(b1, f"{p}_b1"),
                 numpy_helper.from_array(w2, f"{p}_w2"),
                 numpy_helper.from_array(b2, f"{p}_b2"),
                 numpy_helper.from_array(w3, f"{p}_w3"),
                 numpy_helper.from_array(b3, f"{p}_b3"),
                 numpy_helper.from_array(wm, f"{p}_wm"),
                 numpy_helper.from_array(bm, f"{p}_bm")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H, W], ctx.dtype, ctx.layout))


# ── 11. Spatial Attention (CBAM-style) ────────────────────────────────────
class SpatialAttentionCBAM(OTP):
    """ReduceMean+ReduceMax → Concat → Conv → Sigmoid → Mul. CBAM spatial gate."""
    name = "spatial_attention_cbam"
    category = CAT_BRANCH
    target_optimization = "spatial_attention_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "cbam")

        w = self._make_conv_weight(rng, 1, 2, 7)
        b = np.zeros(1, dtype=np.float32)

        avg_o = f"{p}_avg"; max_o = f"{p}_max"
        cat_o = f"{p}_cat"; cv_o = f"{p}_cv"; sig_o = f"{p}_sig"; out = f"{p}_out"
        nodes = [
            helper.make_node("ReduceMean", [input_name], [avg_o],
                             axes=[1], keepdims=1),
            helper.make_node("ReduceMax",  [input_name], [max_o],
                             axes=[1], keepdims=1),
            helper.make_node("Concat",    [avg_o, max_o], [cat_o], axis=1),
            helper.make_node("Conv",      [cat_o, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[7,7], pads=[3]*4, strides=[1,1]),
            helper.make_node("Sigmoid",   [cv_o], [sig_o]),
            helper.make_node("Mul",       [input_name, sig_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 12. FPN Upsample+Add branch ───────────────────────────────────────────
class FPNBranch(OTP):
    """Upsample (nearest) + Add lateral connection: FPN neck pattern."""
    name = "fpn_branch"
    category = CAT_BRANCH
    target_optimization = "upsample_add_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "fpn")

        # Lateral 1×1 conv to match channels, then upsample
        wl = self._make_conv_weight(rng, out_c, C, 1)
        bl = np.zeros(out_c, dtype=np.float32)
        # Top-down path: upsample input then conv
        wu = self._make_conv_weight(rng, out_c, C, 1)
        bu = np.zeros(out_c, dtype=np.float32)
        # Resize scales for both paths: 2× spatial upsample
        scale_f = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        lat_o = f"{p}_lat"; lat_up = f"{p}_lup"
        up_o  = f"{p}_up";  up_cv  = f"{p}_upcv"; out = f"{p}_out"
        nodes = [
            # Lateral branch: conv then upsample → [N, out_c, H*2, W*2]
            helper.make_node("Conv",   [input_name, f"{p}_wl", f"{p}_bl"], [lat_o],
                             kernel_shape=[1,1], pads=[0]*4, strides=[1,1]),
            helper.make_node("Resize", [lat_o, "", f"{p}_scale"], [lat_up],
                             mode="nearest", coordinate_transformation_mode="asymmetric"),
            # Top-down branch: upsample then conv → [N, out_c, H*2, W*2]
            helper.make_node("Resize", [input_name, "", f"{p}_scale"], [up_o],
                             mode="nearest", coordinate_transformation_mode="asymmetric"),
            helper.make_node("Conv",   [up_o, f"{p}_wu", f"{p}_bu"], [up_cv],
                             kernel_shape=[1,1], pads=[0]*4, strides=[1,1]),
            # Add — both at [N, out_c, H*2, W*2]
            helper.make_node("Add",    [lat_up, up_cv], [out]),
        ]
        inits = [numpy_helper.from_array(wl,      f"{p}_wl"),
                 numpy_helper.from_array(bl,      f"{p}_bl"),
                 numpy_helper.from_array(wu,      f"{p}_wu"),
                 numpy_helper.from_array(bu,      f"{p}_bu"),
                 numpy_helper.from_array(scale_f, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H*2, W*2],
                                                 ctx.dtype, ctx.layout))


# ── 13. Dense Residual (multiple skip connections) ────────────────────────
class DenseResidualBlock(OTP):
    """DenseNet-style: x → conv1 → add(x) → conv2 → add(x+h1) → out."""
    name = "dense_residual_block"
    category = CAT_BRANCH
    target_optimization = "dense_skip_merging"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "drb")

        w1 = self._make_conv_weight(rng, C, C, 3)
        w2 = self._make_conv_weight(rng, C, C, 3)
        b1 = np.zeros(C, dtype=np.float32)
        b2 = np.zeros(C, dtype=np.float32)

        cv1 = f"{p}_cv1"; rel1 = f"{p}_r1"; add1 = f"{p}_a1"
        cv2 = f"{p}_cv2"; rel2 = f"{p}_r2"; out  = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w1", f"{p}_b1"], [cv1],
                             kernel_shape=[3,3], pads=[1]*4, strides=[1,1]),
            helper.make_node("Relu", [cv1], [rel1]),
            helper.make_node("Add",  [input_name, rel1], [add1]),
            helper.make_node("Conv", [add1, f"{p}_w2", f"{p}_b2"], [cv2],
                             kernel_shape=[3,3], pads=[1]*4, strides=[1,1]),
            helper.make_node("Relu", [cv2], [rel2]),
            helper.make_node("Add",  [add1, rel2], [out]),
        ]
        inits = [numpy_helper.from_array(w1, f"{p}_w1"),
                 numpy_helper.from_array(b1, f"{p}_b1"),
                 numpy_helper.from_array(w2, f"{p}_w2"),
                 numpy_helper.from_array(b2, f"{p}_b2")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 14. Channel-wise Gating Network ──────────────────────────────────────
class ChannelGatingBranch(OTP):
    """Separate gate branch: gate = Sigmoid(Linear(GAP(x))); out = gate * x."""
    name = "channel_gating_branch"
    category = CAT_BRANCH
    target_optimization = "channel_gate_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "cgb")

        w1 = rng.normal(0, np.sqrt(2.0 / C), (C, max(C // 4, 4))).astype(np.float32)
        w2 = rng.normal(0, np.sqrt(2.0 / max(C//4,4)),
                        (max(C // 4, 4), C)).astype(np.float32)
        b1 = np.zeros(max(C // 4, 4), dtype=np.float32)
        b2 = np.zeros(C, dtype=np.float32)

        gap_o = f"{p}_gap"; fl_o = f"{p}_fl"; mm1 = f"{p}_mm1"
        a1  = f"{p}_a1";   re_o = f"{p}_re"; mm2 = f"{p}_mm2"
        a2  = f"{p}_a2";   sig  = f"{p}_sig"; us = f"{p}_us"; out = f"{p}_out"
        us_axes = np.array([2, 3], dtype=np.int64)
        nodes = [
            helper.make_node("GlobalAveragePool", [input_name], [gap_o]),
            helper.make_node("Flatten",           [gap_o], [fl_o], axis=1),
            helper.make_node("MatMul",            [fl_o, f"{p}_w1"], [mm1]),
            helper.make_node("Add",               [mm1, f"{p}_b1"], [a1]),
            helper.make_node("Relu",              [a1], [re_o]),
            helper.make_node("MatMul",            [re_o, f"{p}_w2"], [mm2]),
            helper.make_node("Add",               [mm2, f"{p}_b2"], [a2]),
            helper.make_node("Sigmoid",           [a2], [sig]),
            helper.make_node("Unsqueeze",         [sig, f"{p}_usax"], [us]),
            helper.make_node("Mul",               [input_name, us], [out]),
        ]
        inits = [numpy_helper.from_array(w1,      f"{p}_w1"),
                 numpy_helper.from_array(b1,      f"{p}_b1"),
                 numpy_helper.from_array(w2,      f"{p}_w2"),
                 numpy_helper.from_array(b2,      f"{p}_b2"),
                 numpy_helper.from_array(us_axes, f"{p}_usax")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 15. MaxPool + AvgPool Fusion branch ──────────────────────────────────
class MaxAvgPoolFusion(OTP):
    """MaxPool and AvgPool in parallel, then Add. Tests parallel pool fusion."""
    name = "max_avg_pool_fusion"
    category = CAT_BRANCH
    target_optimization = "parallel_pool_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H >= 4 and W >= 4

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "mapf")

        mp_o = f"{p}_mp"; ap_o = f"{p}_ap"; out = f"{p}_out"
        H2 = H // 2; W2 = W // 2
        nodes = [
            helper.make_node("MaxPool",     [input_name], [mp_o],
                             kernel_shape=[2,2], strides=[2,2], pads=[0]*4),
            helper.make_node("AveragePool", [input_name], [ap_o],
                             kernel_shape=[2,2], strides=[2,2], pads=[0]*4),
            helper.make_node("Add",         [mp_o, ap_o], [out]),
        ]
        return PatternInstance(self.name, self.category, nodes, [],
                               input_name, out,
                               StructuralContext(4, [N, C, H2, W2], ctx.dtype, ctx.layout))


# ── 16. SE + Residual (combined squeeze-excitation + skip) ───────────────
class SEResidual(OTP):
    """SEBlock output added back to input (SE + residual). Full SE-Net unit."""
    name = "se_residual"
    category = CAT_BRANCH
    target_optimization = "se_residual_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        return ctx.shape[1] >= 8

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        r = max(C // 4, 4)
        p = self._p(node_id, "ser")

        w1 = rng.normal(0, np.sqrt(2.0 / C), (C, r)).astype(np.float32)
        w2 = rng.normal(0, np.sqrt(2.0 / r), (r, C)).astype(np.float32)
        b1 = np.zeros(r, dtype=np.float32)
        b2 = np.zeros(C, dtype=np.float32)

        gap_o = f"{p}_gap"; fl_o = f"{p}_fl"; mm1 = f"{p}_mm1"
        a1   = f"{p}_a1"; re_o = f"{p}_re"; mm2 = f"{p}_mm2"
        a2   = f"{p}_a2"; sig  = f"{p}_sig"
        us   = f"{p}_us"; sc_o = f"{p}_sc"; out = f"{p}_out"
        us_axes = np.array([2, 3], dtype=np.int64)
        nodes = [
            helper.make_node("GlobalAveragePool", [input_name], [gap_o]),
            helper.make_node("Flatten",           [gap_o], [fl_o], axis=1),
            helper.make_node("MatMul",            [fl_o, f"{p}_w1"], [mm1]),
            helper.make_node("Add",               [mm1, f"{p}_b1"], [a1]),
            helper.make_node("Relu",              [a1], [re_o]),
            helper.make_node("MatMul",            [re_o, f"{p}_w2"], [mm2]),
            helper.make_node("Add",               [mm2, f"{p}_b2"], [a2]),
            helper.make_node("Sigmoid",           [a2], [sig]),
            helper.make_node("Unsqueeze",         [sig, f"{p}_usax"], [us]),
            helper.make_node("Mul",               [input_name, us], [sc_o]),
            helper.make_node("Add",               [input_name, sc_o], [out]),
        ]
        inits = [numpy_helper.from_array(w1,      f"{p}_w1"),
                 numpy_helper.from_array(b1,      f"{p}_b1"),
                 numpy_helper.from_array(w2,      f"{p}_w2"),
                 numpy_helper.from_array(b2,      f"{p}_b2"),
                 numpy_helper.from_array(us_axes, f"{p}_usax")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


ALL_BRANCH_PATTERNS = [
    SplitTransformConcat(),
    ResidualAddReLU(),
    ConvBranchAdd(),
    DualPoolAdd(),
    SEBlock(),
    GatedLinearUnit(),
    AddLayerNorm(),
    ConcatConv(),
    # New patterns
    MultiScaleConvBranch(),
    ASPPDilatedBranch(),
    SpatialAttentionCBAM(),
    FPNBranch(),
    DenseResidualBlock(),
    ChannelGatingBranch(),
    MaxAvgPoolFusion(),
    SEResidual(),
]
