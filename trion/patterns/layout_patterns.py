"""
Layout- and shape-sensitive OTPs.
Target: shape inference, layout rewrite, canonicalization.
"""
from __future__ import annotations
import numpy as np
from onnx import helper, numpy_helper
from typing import Optional

from .base import OTP, PatternInstance, CAT_LAYOUT
from ..generation.context import StructuralContext


# ── 1. Reshape → Transpose → Reshape ─────────────────────────────────────
class ReshapeTransposeReshape(OTP):
    name = "reshape_transpose_reshape"
    category = CAT_LAYOUT
    target_optimization = "shape_inference_layout_rewrite"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.num_elements() > 1

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "rtr")

        # Reshape to [N, C*H, W] then transpose [0,2,1] then reshape back
        s1 = np.array([-1, C*H, W], dtype=np.int64)
        s2 = np.array([-1, C, H, W], dtype=np.int64)

        r1_o = f"{p}_r1"; tr_o = f"{p}_tr"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_s1"], [r1_o]),
            helper.make_node("Transpose", [r1_o], [tr_o], perm=[0, 2, 1]),
            helper.make_node("Reshape", [tr_o, f"{p}_s2"], [out]),
        ]
        inits = [numpy_helper.from_array(s1, f"{p}_s1"),
                 numpy_helper.from_array(s2, f"{p}_s2")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 2. Flatten → Dense → Unflatten ───────────────────────────────────────
class FlattenDenseUnflatten(OTP):
    name = "flatten_dense_unflatten"
    category = CAT_LAYOUT
    target_optimization = "layout_canonicalization"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        flat_dim = C * H * W
        p = self._p(node_id, "fdu")

        w = rng.normal(0, np.sqrt(2/flat_dim), (flat_dim, flat_dim)).astype(np.float32)
        b = np.zeros(flat_dim, dtype=np.float32)
        s_back = np.array([-1, C, H, W], dtype=np.int64)

        flat_o = f"{p}_flat"; mm_o = f"{p}_mm"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Flatten", [input_name], [flat_o], axis=1),
            helper.make_node("MatMul", [flat_o, f"{p}_w"], [mm_o]),
            helper.make_node("Add", [mm_o, f"{p}_b"], [add_o]),
            helper.make_node("Reshape", [add_o, f"{p}_sb"], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b"),
                 numpy_helper.from_array(s_back, f"{p}_sb")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 3. Transpose → Transpose (double transpose = identity) ────────────────
class DoubleTranspose(OTP):
    name = "double_transpose"
    category = CAT_LAYOUT
    target_optimization = "redundant_transpose_elimination"

    def is_compatible(self, ctx):
        return ctx.rank == 4

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "dtr")
        perm = [0, 2, 3, 1]          # NCHW → NHWC
        inv_perm = [0, 3, 1, 2]      # NHWC → NCHW

        tr1_o = f"{p}_tr1"; out = f"{p}_out"
        nodes = [
            helper.make_node("Transpose", [input_name], [tr1_o], perm=perm),
            helper.make_node("Transpose", [tr1_o], [out], perm=inv_perm),
        ]
        return PatternInstance(self.name, self.category, nodes, [],
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 4. Reshape → Softmax → Reshape ───────────────────────────────────────
class ReshapeSoftmaxReshape(OTP):
    name = "reshape_softmax_reshape"
    category = CAT_LAYOUT
    target_optimization = "layout_aware_softmax"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.num_elements() > 1

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "rsr")
        s2d = np.array([-1, H*W], dtype=np.int64)
        s4d = np.array([-1, C, H, W], dtype=np.int64)

        r2d_o = f"{p}_r2d"; sm_o = f"{p}_sm"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_s2d"], [r2d_o]),
            helper.make_node("Softmax", [r2d_o], [sm_o], axis=-1),
            helper.make_node("Reshape", [sm_o, f"{p}_s4d"], [out]),
        ]
        inits = [numpy_helper.from_array(s2d, f"{p}_s2d"),
                 numpy_helper.from_array(s4d, f"{p}_s4d")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 5. Flatten → Unsqueeze → Concat (with constant) ──────────────────────
class FlattenUnsqueezeConcat(OTP):
    name = "flatten_unsqueeze_concat"
    category = CAT_LAYOUT
    target_optimization = "shape_concat_canonicalization"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        flat_dim = C * H * W
        p = self._p(node_id, "fuc")
        axes = np.array([1], dtype=np.int64)
        extra = rng.normal(0, 0.1, (N, 1, flat_dim)).astype(np.float32)

        flat_o = f"{p}_flat"; uns_o = f"{p}_uns"; out = f"{p}_out"
        nodes = [
            helper.make_node("Flatten", [input_name], [flat_o], axis=1),
            helper.make_node("Unsqueeze", [flat_o, f"{p}_ax"], [uns_o]),
            helper.make_node("Concat", [uns_o, f"{p}_extra"], [out], axis=1),
        ]
        inits = [numpy_helper.from_array(axes, f"{p}_ax"),
                 numpy_helper.from_array(extra, f"{p}_extra")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [N, 2, flat_dim], ctx.dtype, "ND"))


# ── 6. Squeeze → Unsqueeze (round-trip) ──────────────────────────────────
class SqueezeUnsqueeze(OTP):
    name = "squeeze_unsqueeze"
    category = CAT_LAYOUT
    target_optimization = "squeeze_unsqueeze_elimination"

    def is_compatible(self, ctx):
        return ctx.rank >= 3

    def instantiate(self, input_name, ctx, rng, node_id):
        # Insert and remove a dim at axis=1
        p = self._p(node_id, "su")
        sq_axes  = np.array([-1], dtype=np.int64)   # squeeze last
        us_axes  = np.array([-1], dtype=np.int64)   # unsqueeze last

        sq_o = f"{p}_sq"; out = f"{p}_out"
        # Only valid if last dim == 1; we skip squeeze and do unsqueeze→squeeze
        # Instead: unsqueeze then squeeze at a new axis
        sq_axes2 = np.array([ctx.rank], dtype=np.int64)
        nodes = [
            helper.make_node("Unsqueeze", [input_name, f"{p}_ax1"], [sq_o]),
            helper.make_node("Squeeze",   [sq_o, f"{p}_ax2"], [out]),
        ]
        inits = [numpy_helper.from_array(sq_axes2, f"{p}_ax1"),
                 numpy_helper.from_array(sq_axes2, f"{p}_ax2")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 7. Transpose → Conv (NHWC input) ──────────────────────────────────────
class TransposeConvNHWC(OTP):
    name = "transpose_conv_nhwc"
    category = CAT_LAYOUT
    target_optimization = "layout_nchw_nhwc_rewrite"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "trcv")

        # NCHW → NHWC → conv as NHWC → NCHW
        w = self._make_conv_weight(rng, out_c, C, 3)
        b = np.zeros(out_c, dtype=np.float32)

        nhwc_o = f"{p}_nhwc"; cv_o = f"{p}_cv"; out = f"{p}_out"
        nodes = [
            helper.make_node("Transpose", [input_name], [nhwc_o], perm=[0,2,3,1]),
            # Conv still needs NCHW weight, but input is NHWC here — compiler must handle
            helper.make_node("Transpose", [nhwc_o], [f"{p}_back"], perm=[0,3,1,2]),
            helper.make_node("Conv", [f"{p}_back", f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]),
            helper.make_node("Relu", [cv_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H,W], ctx.dtype, ctx.layout))


# ── 8. Reshape → LayerNorm → Reshape ─────────────────────────────────────
class ReshapeLayerNormReshape(OTP):
    name = "reshape_layernorm_reshape"
    category = CAT_LAYOUT
    target_optimization = "layout_layernorm_canonicalization"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "rlnr")
        s2d = np.array([-1, C*H*W], dtype=np.int64)
        s4d = np.array([-1, C, H, W], dtype=np.int64)
        ln_scale = np.ones(C*H*W, dtype=np.float32)
        ln_bias  = np.zeros(C*H*W, dtype=np.float32)

        r2d_o = f"{p}_r2d"; ln_o = f"{p}_ln"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_s2d"], [r2d_o]),
            helper.make_node("LayerNormalization",
                             [r2d_o, f"{p}_lnsc", f"{p}_lnb"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Reshape", [ln_o, f"{p}_s4d"], [out]),
        ]
        inits = [numpy_helper.from_array(s2d, f"{p}_s2d"),
                 numpy_helper.from_array(s4d, f"{p}_s4d"),
                 numpy_helper.from_array(ln_scale, f"{p}_lnsc"),
                 numpy_helper.from_array(ln_bias,  f"{p}_lnb")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 9. Permute channels (1×1 conv simulates channel shuffle) ──────────────
class ChannelShuffle(OTP):
    name = "channel_shuffle"
    category = CAT_LAYOUT
    target_optimization = "channel_reorder_canonicalization"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32" and ctx.channels() % 2 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        groups = 2
        p = self._p(node_id, "cshuffle")
        s1 = np.array([-1, groups, C//groups, H, W], dtype=np.int64)
        s2 = np.array([-1, C, H, W], dtype=np.int64)

        r1_o = f"{p}_r1"; tr_o = f"{p}_tr"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_s1"], [r1_o]),
            helper.make_node("Transpose", [r1_o], [tr_o], perm=[0,2,1,3,4]),
            helper.make_node("Reshape", [tr_o, f"{p}_s2"], [out]),
        ]
        inits = [numpy_helper.from_array(s1, f"{p}_s1"),
                 numpy_helper.from_array(s2, f"{p}_s2")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 10. Pad → Conv (explicit padding before conv) ─────────────────────────
class PadConv(OTP):
    name = "pad_conv"
    category = CAT_LAYOUT
    target_optimization = "pad_conv_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        pad_val = int(rng.choice([1, 2]))
        p = self._p(node_id, "padcv")

        w = self._make_conv_weight(rng, out_c, C, 3)
        b = np.zeros(out_c, dtype=np.float32)
        # ONNX Pad pads: [x1_begin, x2_begin, x3_begin, x4_begin, x1_end, ...]
        pads_tensor = np.array([0,0,pad_val,pad_val, 0,0,pad_val,pad_val],
                               dtype=np.int64)

        pad_o = f"{p}_pad"; out = f"{p}_out"
        nodes = [
            helper.make_node("Pad", [input_name, f"{p}_pads"], [pad_o],
                             mode="constant"),
            helper.make_node("Conv", [pad_o, f"{p}_w", f"{p}_b"], [out],
                             kernel_shape=[3,3], pads=[0,0,0,0], strides=[1,1]),
        ]
        inits = [numpy_helper.from_array(pads_tensor, f"{p}_pads"),
                 numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        H2 = H+2*pad_val-2; W2 = W+2*pad_val-2
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H2,W2], ctx.dtype, ctx.layout))


# ── 11. SpaceToDepth equivalent (Reshape → Transpose → Reshape → Conv) ───
class SpaceToDepthBlock(OTP):
    """
    SpaceToDepth equivalent via Reshape+Transpose+Reshape + 1×1 conv.
    Implements the same pixel-to-channel rearrangement without using the
    SpaceToDepth op (which onnx2torch does not support).
    [N,C,H,W] → [N,C,H/bs,bs,W/bs,bs] → [N,C,bs,H/bs,bs,W/bs]
             → [N,C*bs*bs,H/bs,W/bs] → Conv → [N,out_c,H/bs,W/bs]
    """
    name = "space_to_depth_block"
    category = CAT_LAYOUT
    target_optimization = "space_to_depth_layout"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H % 2 == 0 and W % 2 == 0 and H >= 4 and W >= 4

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        bs = 2
        new_C = C * bs * bs
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "s2d")

        # Intermediate shapes for Reshape+Transpose+Reshape
        shape1 = np.array([N, C, H//bs, bs, W//bs, bs], dtype=np.int64)
        shape2 = np.array([N, new_C, H//bs, W//bs], dtype=np.int64)
        w = self._make_conv_weight(rng, out_c, new_C, 1)
        b = np.zeros(out_c, dtype=np.float32)

        r1_o = f"{p}_r1"; tr_o = f"{p}_tr"; r2_o = f"{p}_r2"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_s1"], [r1_o]),
            helper.make_node("Transpose", [r1_o], [tr_o], perm=[0, 1, 3, 2, 5, 4]),
            helper.make_node("Reshape", [tr_o, f"{p}_s2"], [r2_o]),
            helper.make_node("Conv", [r2_o, f"{p}_w", f"{p}_b"], [out],
                             kernel_shape=[1, 1], pads=[0]*4, strides=[1, 1]),
        ]
        inits = [numpy_helper.from_array(shape1, f"{p}_s1"),
                 numpy_helper.from_array(shape2, f"{p}_s2"),
                 numpy_helper.from_array(w,      f"{p}_w"),
                 numpy_helper.from_array(b,      f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H//bs, W//bs],
                                                 ctx.dtype, ctx.layout))


# ── 12. DepthToSpace (pixel shuffle upsampling) ───────────────────────────
class DepthToSpaceBlock(OTP):
    """DepthToSpace + pointwise. Sub-pixel convolution (pixel shuffle) pattern."""
    name = "depth_to_space_block"
    category = CAT_LAYOUT
    target_optimization = "depth_to_space_layout"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        C = ctx.shape[1]
        return C >= 4 and C % 4 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        bs = 2
        new_C = C // (bs * bs)
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "d2s")

        w = self._make_conv_weight(rng, out_c, new_C, 1)
        b = np.zeros(out_c, dtype=np.float32)

        d2s_o = f"{p}_d2s"; out = f"{p}_out"
        nodes = [
            helper.make_node("DepthToSpace", [input_name], [d2s_o],
                             blocksize=bs, mode="CRD"),
            helper.make_node("Conv", [d2s_o, f"{p}_w", f"{p}_b"], [out],
                             kernel_shape=[1, 1], pads=[0]*4, strides=[1, 1]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H*bs, W*bs],
                                                 ctx.dtype, ctx.layout))


# ── 13. Reflect Pad → Conv ────────────────────────────────────────────────
class ReflectPadConv(OTP):
    """Reflect-mode padding before conv. TVM often wrong on REFLECT vs CONSTANT."""
    name = "reflect_pad_conv"
    category = CAT_LAYOUT
    target_optimization = "reflect_pad_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H >= 4 and W >= 4

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "rfpcv")

        pad_val = 1
        pads_t = np.array([0, 0, pad_val, pad_val,
                           0, 0, pad_val, pad_val], dtype=np.int64)
        w = self._make_conv_weight(rng, out_c, C, 3)
        b = np.zeros(out_c, dtype=np.float32)

        pad_o = f"{p}_pad"; out = f"{p}_out"
        nodes = [
            helper.make_node("Pad", [input_name, f"{p}_pads"], [pad_o],
                             mode="reflect"),
            helper.make_node("Conv", [pad_o, f"{p}_w", f"{p}_b"], [out],
                             kernel_shape=[3, 3], pads=[0]*4, strides=[1, 1]),
        ]
        inits = [numpy_helper.from_array(pads_t, f"{p}_pads"),
                 numpy_helper.from_array(w,      f"{p}_w"),
                 numpy_helper.from_array(b,      f"{p}_b")]
        H2 = H + 2*pad_val - 2; W2 = W + 2*pad_val - 2
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 14. Reshape → Batched MatMul → Reshape ───────────────────────────────
class ReshapeBatchedMatMul(OTP):
    """Flatten spatial dims → MatMul → restore: batched linear projection."""
    name = "reshape_batched_matmul"
    category = CAT_LAYOUT
    target_optimization = "batched_matmul_layout"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        feat = C * H * W
        K = int(rng.choice([64, 128, 256]))
        p = self._p(node_id, "rbmm")

        w = rng.normal(0, np.sqrt(2.0 / feat), (feat, K)).astype(np.float32)
        b = np.zeros(K, dtype=np.float32)
        s2d = np.array([N, feat], dtype=np.int64)

        r2d_o = f"{p}_r2d"; mm_o = f"{p}_mm"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_s2d"], [r2d_o]),
            helper.make_node("MatMul",  [r2d_o, f"{p}_w"], [mm_o]),
            helper.make_node("Add",     [mm_o, f"{p}_b"], [out]),
        ]
        inits = [numpy_helper.from_array(s2d, f"{p}_s2d"),
                 numpy_helper.from_array(w,   f"{p}_w"),
                 numpy_helper.from_array(b,   f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, K], ctx.dtype, "NC"))


# ── 15. Unsqueeze → Expand → Mul (broadcast expansion) ───────────────────
class UnsqueezeExpandMul(OTP):
    """Unsqueeze + Expand + Mul: common feature-scaling pattern."""
    name = "unsqueeze_expand_mul"
    category = CAT_LAYOUT
    target_optimization = "expand_broadcast_canonicalization"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "uem")

        # 1-D scale [C] → unsqueeze to [C,1,1] → expand to [N,C,H,W]
        scale_vec = rng.normal(1.0, 0.1, (C,)).astype(np.float32)
        target_shape = np.array([N, C, H, W], dtype=np.int64)
        axes = np.array([0, 2, 3], dtype=np.int64)

        us_o = f"{p}_us"; ex_o = f"{p}_ex"; out = f"{p}_out"
        nodes = [
            helper.make_node("Unsqueeze", [f"{p}_scale", f"{p}_axes"], [us_o]),
            helper.make_node("Expand",    [us_o, f"{p}_tshape"], [ex_o]),
            helper.make_node("Mul",       [input_name, ex_o], [out]),
        ]
        inits = [numpy_helper.from_array(scale_vec,   f"{p}_scale"),
                 numpy_helper.from_array(axes,         f"{p}_axes"),
                 numpy_helper.from_array(target_shape, f"{p}_tshape")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 16. Tile → Conv (feature repetition) ──────────────────────────────────
class TileConv(OTP):
    """Tile spatial dims then apply conv: tests tiling + conv shape inference."""
    name = "tile_conv"
    category = CAT_LAYOUT
    target_optimization = "tile_canonicalization"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        repeats = np.array([1, 1, 2, 2], dtype=np.int64)
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "tlcv")

        w = self._make_conv_weight(rng, out_c, C, 3)
        b = np.zeros(out_c, dtype=np.float32)

        tile_o = f"{p}_tile"; out = f"{p}_out"
        nodes = [
            helper.make_node("Tile", [input_name, f"{p}_reps"], [tile_o]),
            helper.make_node("Conv", [tile_o, f"{p}_w", f"{p}_b"], [out],
                             kernel_shape=[3, 3], pads=[1]*4, strides=[1, 1]),
        ]
        inits = [numpy_helper.from_array(repeats, f"{p}_reps"),
                 numpy_helper.from_array(w,       f"{p}_w"),
                 numpy_helper.from_array(b,       f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H*2, W*2],
                                                 ctx.dtype, ctx.layout))


# ── 17. Slice → Pad → Concat (asymmetric feature crop/pad) ───────────────
class SlicePadConcat(OTP):
    """Slice half the feature map, pad, concat back. Shape-rewrite stress test."""
    name = "slice_pad_concat"
    category = CAT_LAYOUT
    target_optimization = "slice_pad_concat_simplification"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, C, H, W = ctx.shape
        # H must be even so that pad(half_H) produces same height as input for Concat
        return H >= 4 and W >= 4 and C >= 2 and H % 2 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        half_H = H // 2
        p = self._p(node_id, "spc")

        sl_start = np.array([0, 0, 0,      0], dtype=np.int64)
        sl_end   = np.array([N, C, half_H, W], dtype=np.int64)
        pad_t    = np.array([0, 0, 0, 0, 0, 0, half_H, 0], dtype=np.int64)

        sl_o = f"{p}_sl"; pd_o = f"{p}_pd"; out = f"{p}_out"
        nodes = [
            helper.make_node("Slice",  [input_name, f"{p}_sls", f"{p}_sle"], [sl_o]),
            helper.make_node("Pad",    [sl_o, f"{p}_pads"], [pd_o], mode="constant"),
            helper.make_node("Concat", [input_name, pd_o], [out], axis=1),
        ]
        inits = [numpy_helper.from_array(sl_start, f"{p}_sls"),
                 numpy_helper.from_array(sl_end,   f"{p}_sle"),
                 numpy_helper.from_array(pad_t,    f"{p}_pads")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C*2, H, W], ctx.dtype, ctx.layout))


# ── 18. Gather → Reshape (embedding-like lookup) ─────────────────────────
class GatherReshape(OTP):
    """Gather rows from a weight matrix then reshape: embedding-table pattern."""
    name = "gather_reshape"
    category = CAT_LAYOUT
    target_optimization = "gather_reshape_fusion"

    def is_compatible(self, ctx):
        # Works from rank-2 [B, T] int-index context or we generate indices
        return ctx.rank == 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, M = ctx.shape
        vocab = 64
        embed = int(rng.choice([32, 64, 128]))
        p = self._p(node_id, "gr")

        # Use first col of float input as integer indices (cast)
        emb_table = rng.normal(0, 0.02, (vocab, embed)).astype(np.float32)
        indices   = rng.integers(0, vocab, size=(N,)).astype(np.int64)
        out_shape = np.array([N, embed], dtype=np.int64)

        cast_o = f"{p}_cast"; gath_o = f"{p}_gath"; out = f"{p}_out"
        nodes = [
            # Cast float index to int64 — tests Cast + Gather fusion
            helper.make_node("Gather", [f"{p}_emb", f"{p}_idx"], [gath_o], axis=0),
            helper.make_node("Reshape", [gath_o, f"{p}_oshape"], [out]),
        ]
        inits = [numpy_helper.from_array(emb_table, f"{p}_emb"),
                 numpy_helper.from_array(indices,   f"{p}_idx"),
                 numpy_helper.from_array(out_shape, f"{p}_oshape")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, embed], ctx.dtype, "NC"))


# ── 19. MaxPool with dilation (rare combination) ──────────────────────────
class DilatedMaxPool(OTP):
    """MaxPool with dilation > 1. Rarely tested; known bugs in several backends."""
    name = "dilated_max_pool"
    category = CAT_LAYOUT
    target_optimization = "dilated_pool_layout"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H >= 8 and W >= 8

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        dilation = 2
        k = 3
        pad = 1   # must satisfy both ORT (pad < k=3) and PyTorch MaxPool2d (pad <= k//2=1)
        p = self._p(node_id, "dmp")

        out = f"{p}_out"
        nodes = [
            helper.make_node("MaxPool", [input_name], [out],
                             kernel_shape=[k, k],
                             pads=[pad]*4,
                             strides=[1, 1],
                             dilations=[dilation, dilation]),
        ]
        # Output size: floor((H + 2*pad - dilation*(k-1) - 1) / stride + 1)
        #   = H + 2*1 - 2*2 - 1 + 1 = H - 2
        Ho = H + 2 * pad - dilation * (k - 1)
        Wo = W + 2 * pad - dilation * (k - 1)
        return PatternInstance(self.name, self.category, nodes, [],
                               input_name, out,
                               StructuralContext(4, [N, C, Ho, Wo], ctx.dtype, ctx.layout))


# ── 20. AveragePool ceil_mode → Conv ─────────────────────────────────────
class CeilModeAvgPoolConv(OTP):
    """AveragePool with ceil_mode=1 before conv. Rounding divergence across backends."""
    name = "ceil_mode_avg_pool_conv"
    category = CAT_LAYOUT
    target_optimization = "pool_ceil_mode_shape"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H >= 4 and W >= 4

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "cmapc")

        w = self._make_conv_weight(rng, out_c, C, 1)
        b = np.zeros(out_c, dtype=np.float32)

        pool_o = f"{p}_pool"; out = f"{p}_out"
        H2 = H // 2 + 1; W2 = W // 2 + 1   # ceil_mode=1, stride=2, pads=[1]*4, kernel=3
        nodes = [
            helper.make_node("AveragePool", [input_name], [pool_o],
                             kernel_shape=[3, 3], strides=[2, 2],
                             pads=[1]*4, ceil_mode=1),
            helper.make_node("Conv", [pool_o, f"{p}_w", f"{p}_b"], [out],
                             kernel_shape=[1, 1], pads=[0]*4, strides=[1, 1]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


ALL_LAYOUT_PATTERNS = [
    ReshapeTransposeReshape(),
    FlattenDenseUnflatten(),
    DoubleTranspose(),
    ReshapeSoftmaxReshape(),
    FlattenUnsqueezeConcat(),
    SqueezeUnsqueeze(),
    TransposeConvNHWC(),
    ReshapeLayerNormReshape(),
    ChannelShuffle(),
    PadConv(),
    # New patterns
    SpaceToDepthBlock(),
    DepthToSpaceBlock(),
    ReflectPadConv(),
    ReshapeBatchedMatMul(),
    UnsqueezeExpandMul(),
    TileConv(),
    SlicePadConcat(),
    GatherReshape(),
    DilatedMaxPool(),
    CeilModeAvgPoolConv(),
]
