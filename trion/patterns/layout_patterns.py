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
]
