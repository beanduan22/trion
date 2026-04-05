"""
Fusion-oriented OTPs.
Target: operator fusion, kernel merging, fused-kernel correctness.
"""
from __future__ import annotations
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from typing import Optional

from .base import OTP, PatternInstance, CAT_FUSION
from ..generation.context import StructuralContext


def _bn_params(rng, channels, prefix):
    """Create BatchNorm scale/bias/mean/var initializers."""
    names = [f"{prefix}_bn_scale", f"{prefix}_bn_bias",
             f"{prefix}_bn_mean", f"{prefix}_bn_var"]
    arrays = [
        np.ones(channels, dtype=np.float32),
        np.zeros(channels, dtype=np.float32),
        np.zeros(channels, dtype=np.float32),
        np.ones(channels, dtype=np.float32),
    ]
    return names, [numpy_helper.from_array(a, n) for n, a in zip(names, arrays)]


# ── 1. Conv → BN → ReLU ───────────────────────────────────────────────────
class ConvBNReLU(OTP):
    name = "conv_bn_relu"
    category = CAT_FUSION
    target_optimization = "conv_bn_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3, 5]))
        pad = k // 2
        stride = int(rng.choice([1, 2]))
        p = self._p(node_id, "cbrelu")

        w = self._make_conv_weight(rng, out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(rng, out_c, p)

        conv_out = f"{p}_cv"; bn_out = f"{p}_bn"; out = f"{p}_out"

        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [conv_out],
                             name=f"{p}_conv",
                             kernel_shape=[k, k], pads=[pad]*4, strides=[stride]*2),
            helper.make_node("BatchNormalization",
                             [conv_out] + bn_names, [bn_out],
                             name=f"{p}_bn", epsilon=1e-5),
            helper.make_node("Relu", [bn_out], [out], name=f"{p}_relu"),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits

        H2 = (H + 2*pad - k) // stride + 1
        W2 = (W + 2*pad - k) // stride + 1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 2. Conv → BN → LeakyReLU ──────────────────────────────────────────────
class ConvBNLeakyReLU(OTP):
    name = "conv_bn_leakyrelu"
    category = CAT_FUSION
    target_optimization = "conv_bn_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3]))
        pad = k // 2
        stride = int(rng.choice([1, 2]))
        alpha = float(rng.uniform(0.01, 0.3))
        p = self._p(node_id, "cblrelu")

        w = self._make_conv_weight(rng, out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(rng, out_c, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k,k], pads=[pad]*4, strides=[stride]*2),
            helper.make_node("BatchNormalization", [cv_o]+bn_names, [bn_o],
                             epsilon=1e-5),
            helper.make_node("LeakyRelu", [bn_o], [out], alpha=alpha),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits

        H2 = (H+2*pad-k)//stride+1; W2 = (W+2*pad-k)//stride+1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H2,W2], ctx.dtype, ctx.layout))


# ── 3. Conv → BN → Sigmoid ────────────────────────────────────────────────
class ConvBNSigmoid(OTP):
    name = "conv_bn_sigmoid"
    category = CAT_FUSION
    target_optimization = "conv_bn_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3]))
        pad = k // 2
        p = self._p(node_id, "cbsig")

        w = self._make_conv_weight(rng, out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(rng, out_c, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k,k], pads=[pad]*4, strides=[1,1]),
            helper.make_node("BatchNormalization", [cv_o]+bn_names, [bn_o],
                             epsilon=1e-5),
            helper.make_node("Sigmoid", [bn_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        H2 = H+2*pad-k+1; W2 = W+2*pad-k+1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H2,W2], ctx.dtype, ctx.layout))


# ── 4. Conv → Add → ReLU (residual shortcut) ──────────────────────────────
class ConvAddReLU(OTP):
    name = "conv_add_relu"
    category = CAT_FUSION
    target_optimization = "residual_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        # Keep same channels so residual add is trivial
        k = 1
        p = self._p(node_id, "carelu")

        w = self._make_conv_weight(rng, C, C, k)
        b = np.zeros(C, dtype=np.float32)
        bias_name = f"{p}_bias"
        bias = rng.normal(0, 0.01, (1, C, 1, 1)).astype(np.float32)

        cv_o = f"{p}_cv"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[1,1], pads=[0]*4, strides=[1,1]),
            helper.make_node("Add", [cv_o, bias_name], [add_o]),
            helper.make_node("Relu", [add_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b"),
                 numpy_helper.from_array(bias, bias_name)]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H,W], ctx.dtype, ctx.layout))


# ── 5. MatMul → BiasAdd → ReLU ────────────────────────────────────────────
class MatMulBiasReLU(OTP):
    name = "matmul_bias_relu"
    category = CAT_FUSION
    target_optimization = "gemm_relu_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, M = ctx.shape
        K = int(rng.choice([64, 128, 256, 512]))
        p = self._p(node_id, "mmbrelu")

        w = rng.normal(0, np.sqrt(2/M), (M, K)).astype(np.float32)
        b = np.zeros(K, dtype=np.float32)

        mm_o = f"{p}_mm"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("MatMul", [input_name, f"{p}_w"], [mm_o]),
            helper.make_node("Add", [mm_o, f"{p}_b"], [add_o]),
            helper.make_node("Relu", [add_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, K], ctx.dtype, ctx.layout))


# ── 6. MatMul → BiasAdd → GELU ────────────────────────────────────────────
class MatMulBiasGELU(OTP):
    name = "matmul_bias_gelu"
    category = CAT_FUSION
    target_optimization = "gemm_gelu_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, M = ctx.shape
        K = int(rng.choice([64, 128, 256]))
        p = self._p(node_id, "mmbgelu")

        w = rng.normal(0, np.sqrt(2/M), (M, K)).astype(np.float32)
        b = np.zeros(K, dtype=np.float32)

        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt2 = np.array([np.sqrt(2.0)], dtype=np.float32)
        half  = np.array([0.5], dtype=np.float32)
        one   = np.array([1.0], dtype=np.float32)

        mm_o = f"{p}_mm"; add_o = f"{p}_add"
        div_o = f"{p}_div"; erf_o = f"{p}_erf"; erf1_o = f"{p}_erf1"
        mul1_o = f"{p}_mul1"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul", [input_name, f"{p}_w"], [mm_o]),
            helper.make_node("Add", [mm_o, f"{p}_b"], [add_o]),
            helper.make_node("Div", [add_o, f"{p}_sqrt2"], [div_o]),
            helper.make_node("Erf", [div_o], [erf_o]),
            helper.make_node("Add", [erf_o, f"{p}_one"], [erf1_o]),
            helper.make_node("Mul", [add_o, f"{p}_half"], [mul1_o]),
            helper.make_node("Mul", [mul1_o, erf1_o], [out]),
        ]
        inits = [
            numpy_helper.from_array(w, f"{p}_w"),
            numpy_helper.from_array(b, f"{p}_b"),
            numpy_helper.from_array(sqrt2, f"{p}_sqrt2"),
            numpy_helper.from_array(half,  f"{p}_half"),
            numpy_helper.from_array(one,   f"{p}_one"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, K], ctx.dtype, ctx.layout))


# ── 7. DepthwiseConv → BN → ReLU ──────────────────────────────────────────
class DepthwiseConvBNReLU(OTP):
    name = "depthwise_conv_bn_relu"
    category = CAT_FUSION
    target_optimization = "depthwise_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32" and ctx.channels() is not None

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        k = int(rng.choice([3, 5]))
        pad = k // 2
        p = self._p(node_id, "dwcbrelu")

        # Depthwise: group = C, weight shape [C, 1, k, k]
        w = rng.normal(0, 0.1, (C, 1, k, k)).astype(np.float32)
        b = np.zeros(C, dtype=np.float32)
        bn_names, bn_inits = _bn_params(rng, C, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k,k], pads=[pad]*4, strides=[1,1], group=C),
            helper.make_node("BatchNormalization", [cv_o]+bn_names, [bn_o],
                             epsilon=1e-5),
            helper.make_node("Relu", [bn_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        H2 = H+2*pad-k+1; W2 = W+2*pad-k+1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,C,H2,W2], ctx.dtype, ctx.layout))


# ── 8. Conv → ReLU → Conv → BN (two-layer fused block) ───────────────────
class ConvReLUConvBN(OTP):
    name = "conv_relu_conv_bn"
    category = CAT_FUSION
    target_optimization = "multi_op_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        mid_c = self._rand_channels(rng)
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "crcrbn")

        w1 = self._make_conv_weight(rng, mid_c, C, 3)
        b1 = np.zeros(mid_c, dtype=np.float32)
        w2 = self._make_conv_weight(rng, out_c, mid_c, 1)
        b2 = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(rng, out_c, p)

        cv1_o = f"{p}_cv1"; relu_o = f"{p}_relu"
        cv2_o = f"{p}_cv2"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w1", f"{p}_b1"], [cv1_o],
                             kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]),
            helper.make_node("Relu", [cv1_o], [relu_o]),
            helper.make_node("Conv", [relu_o, f"{p}_w2", f"{p}_b2"], [cv2_o],
                             kernel_shape=[1,1], pads=[0,0,0,0], strides=[1,1]),
            helper.make_node("BatchNormalization", [cv2_o]+bn_names, [out],
                             epsilon=1e-5),
        ]
        inits = ([numpy_helper.from_array(w1, f"{p}_w1"),
                  numpy_helper.from_array(b1, f"{p}_b1"),
                  numpy_helper.from_array(w2, f"{p}_w2"),
                  numpy_helper.from_array(b2, f"{p}_b2")] + bn_inits)
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H,W], ctx.dtype, ctx.layout))


# ── 9. Linear → LayerNorm → GELU ──────────────────────────────────────────
class LinearLayerNormGELU(OTP):
    name = "linear_layernorm_gelu"
    category = CAT_FUSION
    target_optimization = "transformer_block_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, M = ctx.shape
        K = int(rng.choice([64, 128, 256, 512]))
        p = self._p(node_id, "llng")

        w = rng.normal(0, np.sqrt(2/M), (M, K)).astype(np.float32)
        b = np.zeros(K, dtype=np.float32)
        ln_scale = np.ones(K, dtype=np.float32)
        ln_bias  = np.zeros(K, dtype=np.float32)
        sqrt2 = np.array([np.sqrt(2.0)], dtype=np.float32)
        half  = np.array([0.5], dtype=np.float32)
        one   = np.array([1.0], dtype=np.float32)

        mm_o = f"{p}_mm"; ln_o = f"{p}_ln"
        div_o = f"{p}_div"; erf_o = f"{p}_erf"; erf1_o = f"{p}_erf1"
        m1_o  = f"{p}_m1"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul", [input_name, f"{p}_w"], [mm_o]),
            helper.make_node("Add", [mm_o, f"{p}_b"], [f"{p}_mmadd"]),
            helper.make_node("LayerNormalization",
                             [f"{p}_mmadd", f"{p}_lnsc", f"{p}_lnb"], [ln_o],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Div",  [ln_o, f"{p}_sqrt2"], [div_o]),
            helper.make_node("Erf",  [div_o], [erf_o]),
            helper.make_node("Add",  [erf_o, f"{p}_one"], [erf1_o]),
            helper.make_node("Mul",  [ln_o, f"{p}_half"], [m1_o]),
            helper.make_node("Mul",  [m1_o, erf1_o], [out]),
        ]
        inits = [
            numpy_helper.from_array(w,        f"{p}_w"),
            numpy_helper.from_array(b,        f"{p}_b"),
            numpy_helper.from_array(ln_scale, f"{p}_lnsc"),
            numpy_helper.from_array(ln_bias,  f"{p}_lnb"),
            numpy_helper.from_array(sqrt2,    f"{p}_sqrt2"),
            numpy_helper.from_array(half,     f"{p}_half"),
            numpy_helper.from_array(one,      f"{p}_one"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, K], ctx.dtype, ctx.layout))


# ── 10. GlobalAvgPool → Flatten → Linear (classification head) ────────────
class GlobalAvgPoolLinear(OTP):
    name = "gap_linear"
    category = CAT_FUSION
    target_optimization = "gap_linear_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        K = int(rng.choice([64, 128, 256]))
        p = self._p(node_id, "gaplin")

        w = rng.normal(0, np.sqrt(2/C), (C, K)).astype(np.float32)
        b = np.zeros(K, dtype=np.float32)

        gap_o = f"{p}_gap"; flat_o = f"{p}_flat"; out = f"{p}_out"
        nodes = [
            helper.make_node("GlobalAveragePool", [input_name], [gap_o]),
            helper.make_node("Flatten", [gap_o], [flat_o], axis=1),
            helper.make_node("MatMul", [flat_o, f"{p}_w"], [f"{p}_mm"]),
            helper.make_node("Add", [f"{p}_mm", f"{p}_b"], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, K], ctx.dtype, "NC"))


# ── 11. Conv → BN → ReLU → MaxPool ───────────────────────────────────────
class ConvBNReLUMaxPool(OTP):
    name = "conv_bn_relu_maxpool"
    category = CAT_FUSION
    target_optimization = "conv_pool_fusion"

    def is_compatible(self, ctx):
        return (ctx.rank == 4 and ctx.dtype == "float32"
                and ctx.shape[2] >= 4 and ctx.shape[3] >= 4)

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = 3; pad = 1
        p = self._p(node_id, "cbrmp")

        w = self._make_conv_weight(rng, out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(rng, out_c, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; relu_o = f"{p}_relu"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k,k], pads=[pad]*4, strides=[1,1]),
            helper.make_node("BatchNormalization", [cv_o]+bn_names, [bn_o],
                             epsilon=1e-5),
            helper.make_node("Relu", [bn_o], [relu_o]),
            helper.make_node("MaxPool", [relu_o], [out],
                             kernel_shape=[2,2], strides=[2,2], pads=[0,0,0,0]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        H2 = (H+2*pad-k+1)//2; W2 = (W+2*pad-k+1)//2
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N,out_c,H2,W2], ctx.dtype, ctx.layout))


# ── 12. MatMul → BiasAdd → Sigmoid (gating) ───────────────────────────────
class MatMulBiasSigmoid(OTP):
    name = "matmul_bias_sigmoid"
    category = CAT_FUSION
    target_optimization = "gemm_sigmoid_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, M = ctx.shape
        K = int(rng.choice([64, 128, 256]))
        p = self._p(node_id, "mmbsig")

        w = rng.normal(0, np.sqrt(2/M), (M, K)).astype(np.float32)
        b = np.zeros(K, dtype=np.float32)

        mm_o = f"{p}_mm"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("MatMul", [input_name, f"{p}_w"], [mm_o]),
            helper.make_node("Add", [mm_o, f"{p}_b"], [add_o]),
            helper.make_node("Sigmoid", [add_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, K], ctx.dtype, ctx.layout))


ALL_FUSION_PATTERNS = [
    ConvBNReLU(),
    ConvBNLeakyReLU(),
    ConvBNSigmoid(),
    ConvAddReLU(),
    MatMulBiasReLU(),
    MatMulBiasGELU(),
    DepthwiseConvBNReLU(),
    ConvReLUConvBN(),
    LinearLayerNormGELU(),
    GlobalAvgPoolLinear(),
    ConvBNReLUMaxPool(),
    MatMulBiasSigmoid(),
]
