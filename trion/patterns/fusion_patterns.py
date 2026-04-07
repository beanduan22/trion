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


def _bn_params(channels, prefix):
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

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

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
        alpha = 0.1
        p = self._p(node_id, "cblrelu")

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

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

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

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

        w = self._make_conv_weight(C, C, k)
        b = np.zeros(C, dtype=np.float32)
        bias_name = f"{p}_bias"
        bias = np.zeros((1, C, 1, 1), dtype=np.float32)

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

        w = self._make_linear_weight(K, M)
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

        w = self._make_linear_weight(K, M)
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
        fan_in = k * k
        _dw_val = np.sqrt(2.0 / fan_in)
        w = np.full((C, 1, k, k), _dw_val, dtype=np.float32)
        w[1::2] = -_dw_val
        b = np.zeros(C, dtype=np.float32)
        bn_names, bn_inits = _bn_params(C, p)

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

        w1 = self._make_conv_weight(mid_c, C, 3)
        b1 = np.zeros(mid_c, dtype=np.float32)
        w2 = self._make_conv_weight(out_c, mid_c, 1)
        b2 = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

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

        w = self._make_linear_weight(K, M)
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

        w = self._make_linear_weight(K, C)
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

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

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

        w = self._make_linear_weight(K, M)
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


# ── 13. ConvTranspose → BN → ReLU (deconvolution) ────────────────────────
class ConvTransposeBNReLU(OTP):
    """Deconvolution + BN + ReLU. Known shape-inference bugs in TVM/TRT."""
    name = "conv_transpose_bn_relu"
    category = CAT_FUSION
    target_optimization = "deconv_bn_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([2, 4]))
        stride = k   # standard upsample factor
        p = self._p(node_id, "ctbnr")

        # ConvTranspose weight: [C_in, C_out, k, k]
        _ct_fan = C * k * k
        _ct_val = np.sqrt(2.0 / _ct_fan)
        w = np.full((C, out_c, k, k), _ct_val, dtype=np.float32)
        w[1::2] = -_ct_val
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

        ct_o = f"{p}_ct"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("ConvTranspose",
                             [input_name, f"{p}_w", f"{p}_b"], [ct_o],
                             kernel_shape=[k, k], strides=[stride, stride]),
            helper.make_node("BatchNormalization",
                             [ct_o] + bn_names, [bn_o], epsilon=1e-5),
            helper.make_node("Relu", [bn_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        H2 = (H - 1) * stride + k
        W2 = (W - 1) * stride + k
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 14. Dilated Conv → BN → ReLU ─────────────────────────────────────────
class DilatedConvBNReLU(OTP):
    """Atrous/dilated convolution. dilation > 1 triggers TVM shape bugs."""
    name = "dilated_conv_bn_relu"
    category = CAT_FUSION
    target_optimization = "dilated_conv_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H >= 8 and W >= 8   # need room for dilation

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        dilation = int(rng.choice([2, 3, 4]))
        k = 3
        pad = dilation   # same-output padding for dilated conv
        p = self._p(node_id, "dcbnr")

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv",
                             [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k, k],
                             pads=[pad]*4,
                             strides=[1, 1],
                             dilations=[dilation, dilation]),
            helper.make_node("BatchNormalization",
                             [cv_o] + bn_names, [bn_o], epsilon=1e-5),
            helper.make_node("Relu", [bn_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        # effective kernel = k + (k-1)*(dilation-1); with pad=dilation: H_out=H
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H, W],
                                                 ctx.dtype, ctx.layout))


# ── 15. Conv → BN → SiLU (Swish) ─────────────────────────────────────────
class ConvBNSiLU(OTP):
    """Conv+BN+SiLU: EfficientNet backbone pattern. Tests Swish fusion."""
    name = "conv_bn_silu"
    category = CAT_FUSION
    target_optimization = "conv_bn_swish_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3]))
        pad = k // 2
        p = self._p(node_id, "cbsilu")

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"
        sig_o = f"{p}_sig"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k, k], pads=[pad]*4, strides=[1, 1]),
            helper.make_node("BatchNormalization",
                             [cv_o] + bn_names, [bn_o], epsilon=1e-5),
            helper.make_node("Sigmoid", [bn_o], [sig_o]),
            helper.make_node("Mul",     [bn_o, sig_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        H2 = H + 2*pad - k + 1; W2 = W + 2*pad - k + 1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 16. Conv → BN → Hardswish ────────────────────────────────────────────
class ConvBNHardswish(OTP):
    """MobileNetV3 head pattern. TRT often has Hardswish precision bugs."""
    name = "conv_bn_hardswish"
    category = CAT_FUSION
    target_optimization = "conv_bn_hardswish_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3]))
        pad = k // 2
        p = self._p(node_id, "cbhs")

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)
        three  = np.array([3.0], dtype=np.float32)
        six    = np.array([6.0], dtype=np.float32)
        zero   = np.array([0.0], dtype=np.float32)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"
        add3 = f"{p}_a3"; clip_o = f"{p}_cl"; div6 = f"{p}_d6"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k, k], pads=[pad]*4, strides=[1, 1]),
            helper.make_node("BatchNormalization",
                             [cv_o] + bn_names, [bn_o], epsilon=1e-5),
            # hardswish(x) = x * clip(x+3, 0, 6) / 6
            helper.make_node("Add",  [bn_o, f"{p}_three"], [add3]),
            helper.make_node("Clip", [add3, f"{p}_zero", f"{p}_six"], [clip_o]),
            helper.make_node("Mul",  [bn_o, clip_o], [div6]),
            helper.make_node("Div",  [div6, f"{p}_six"], [out]),
        ]
        inits = ([numpy_helper.from_array(w, f"{p}_w"),
                  numpy_helper.from_array(b, f"{p}_b")]
                 + bn_inits
                 + [numpy_helper.from_array(three, f"{p}_three"),
                    numpy_helper.from_array(six,   f"{p}_six"),
                    numpy_helper.from_array(zero,  f"{p}_zero")])
        H2 = H + 2*pad - k + 1; W2 = W + 2*pad - k + 1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 17. Conv → BN → ReLU6 (Clip) ─────────────────────────────────────────
class ConvBNReLU6(OTP):
    """MobileNet-v1/v2 activation (Clip 0→6). Tests Clip-fusion pass."""
    name = "conv_bn_relu6"
    category = CAT_FUSION
    target_optimization = "conv_bn_clip_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3]))
        pad = k // 2
        p = self._p(node_id, "cbrelu6")

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)
        zero = np.array([0.0], dtype=np.float32)
        six  = np.array([6.0], dtype=np.float32)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k, k], pads=[pad]*4, strides=[1, 1]),
            helper.make_node("BatchNormalization",
                             [cv_o] + bn_names, [bn_o], epsilon=1e-5),
            helper.make_node("Clip", [bn_o, f"{p}_zero", f"{p}_six"], [out]),
        ]
        inits = ([numpy_helper.from_array(w, f"{p}_w"),
                  numpy_helper.from_array(b, f"{p}_b")]
                 + bn_inits
                 + [numpy_helper.from_array(zero, f"{p}_zero"),
                    numpy_helper.from_array(six,  f"{p}_six")])
        H2 = H + 2*pad - k + 1; W2 = W + 2*pad - k + 1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 18. Grouped Conv → BN → ReLU ─────────────────────────────────────────
class GroupedConvBNReLU(OTP):
    """Group convolution (group > 1). Group handling bugs in TVM + TRT."""
    name = "grouped_conv_bn_relu"
    category = CAT_FUSION
    target_optimization = "grouped_conv_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        return ctx.shape[1] >= 8 and ctx.shape[1] % 4 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        groups = 4
        out_c = C   # grouped conv preserving channels is common
        k = 3; pad = 1
        p = self._p(node_id, "gcbnr")

        # weight shape: [out_c, in_c/groups, k, k]
        _gc_fan = C // groups * k * k
        _gc_val = np.sqrt(2.0 / _gc_fan)
        w = np.full((out_c, C // groups, k, k), _gc_val, dtype=np.float32)
        w[1::2] = -_gc_val
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv",
                             [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k, k], pads=[pad]*4,
                             strides=[1, 1], group=groups),
            helper.make_node("BatchNormalization",
                             [cv_o] + bn_names, [bn_o], epsilon=1e-5),
            helper.make_node("Relu", [bn_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H, W],
                                                 ctx.dtype, ctx.layout))


# ── 19. Conv + Asymmetric Padding → BN ───────────────────────────────────
class ConvAsymPadBN(OTP):
    """
    Asymmetric spatial padding (top≠bottom, left≠right) before 3×3 Conv + BN.
    Exercises pad-canonicalization in compilers.

    Uses Pad node with [0,0,1,0,0,0,0,1] (pad H_start=1, W_end=1), then 3×3
    Conv with no additional padding so total per-dimension pad is asymmetric.
    This is legal ONNX and legal PyTorch (external Pad then Conv with pads=0).
    """
    name = "conv_asym_pad_bn"
    category = CAT_FUSION
    target_optimization = "asymmetric_pad_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, _, H, W = ctx.shape
        return H >= 4 and W >= 4

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        p = self._p(node_id, "capbn")

        w = self._make_conv_weight(out_c, C, 3)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

        # Pad 1 on H_start and 1 on W_end only (asymmetric).
        # ONNX pads layout for 4-D: [n_b,c_b,h_b,w_b, n_e,c_e,h_e,w_e]
        # → H_start=1, W_end=1, all others=0
        pads_t = np.array([0, 0, 1, 0, 0, 0, 0, 1], dtype=np.int64)

        pad_o = f"{p}_pad"; cv_o = f"{p}_cv"; out = f"{p}_out"
        nodes = [
            helper.make_node("Pad", [input_name, f"{p}_pads"], [pad_o],
                             mode="constant"),
            helper.make_node("Conv", [pad_o, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[3, 3], pads=[0]*4, strides=[1, 1]),
            helper.make_node("BatchNormalization",
                             [cv_o] + bn_names, [out], epsilon=1e-5),
        ]
        inits = ([numpy_helper.from_array(pads_t, f"{p}_pads"),
                  numpy_helper.from_array(w,      f"{p}_w"),
                  numpy_helper.from_array(b,      f"{p}_b")]
                 + bn_inits)
        # Output: H+1(pad_h_start) - 2(kernel) = H-1
        #         W+1(pad_w_end)   - 2(kernel) = W-1
        H2 = max(H - 1, 1)
        W2 = max(W - 1, 1)
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 20. Conv → BN → ELU ──────────────────────────────────────────────────
class ConvBNELU(OTP):
    """ELU activation (exp-based). Known precision divergence across backends."""
    name = "conv_bn_elu"
    category = CAT_FUSION
    target_optimization = "conv_bn_elu_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3]))
        pad = k // 2
        alpha = 1.0
        p = self._p(node_id, "cbelu")

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        bn_names, bn_inits = _bn_params(out_c, p)

        cv_o = f"{p}_cv"; bn_o = f"{p}_bn"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k, k], pads=[pad]*4, strides=[1, 1]),
            helper.make_node("BatchNormalization",
                             [cv_o] + bn_names, [bn_o], epsilon=1e-5),
            helper.make_node("Elu", [bn_o], [out], alpha=alpha),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")] + bn_inits
        H2 = H + 2*pad - k + 1; W2 = W + 2*pad - k + 1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 21. Pointwise + Depthwise block (MobileNet inverted residual core) ────
class PointwiseDWBlock(OTP):
    """1×1 expand → DW 3×3 → 1×1 project. Core MobileNet-v2 block."""
    name = "pointwise_dw_block"
    category = CAT_FUSION
    target_optimization = "mobilenet_block_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        expand = max(C * 2, 32)
        p = self._p(node_id, "pwdw")

        w_pw  = self._make_conv_weight(expand, C, 1)
        b_pw  = np.zeros(expand, dtype=np.float32)
        _dw_fan = 3 * 3
        _dw_v = np.sqrt(2.0 / _dw_fan)
        w_dw  = np.full((expand, 1, 3, 3), _dw_v, dtype=np.float32)
        w_dw[1::2] = -_dw_v
        b_dw  = np.zeros(expand, dtype=np.float32)
        w_proj = self._make_conv_weight(C, expand, 1)
        b_proj = np.zeros(C, dtype=np.float32)
        bn1_names, bn1_inits = _bn_params(expand, f"{p}_bn1")
        bn2_names, bn2_inits = _bn_params(expand, f"{p}_bn2")

        pw_o = f"{p}_pw"; bn1_o = f"{p}_bn1o"; relu1 = f"{p}_relu1"
        dw_o = f"{p}_dw"; bn2_o = f"{p}_bn2o"; relu2 = f"{p}_relu2"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_wpw", f"{p}_bpw"], [pw_o],
                             kernel_shape=[1, 1], pads=[0]*4, strides=[1, 1]),
            helper.make_node("BatchNormalization", [pw_o] + bn1_names, [bn1_o], epsilon=1e-5),
            helper.make_node("Relu", [bn1_o], [relu1]),
            helper.make_node("Conv", [relu1, f"{p}_wdw", f"{p}_bdw"], [dw_o],
                             kernel_shape=[3, 3], pads=[1]*4, strides=[1, 1], group=expand),
            helper.make_node("BatchNormalization", [dw_o] + bn2_names, [bn2_o], epsilon=1e-5),
            helper.make_node("Relu", [bn2_o], [relu2]),
            helper.make_node("Conv", [relu2, f"{p}_wproj", f"{p}_bproj"], [out],
                             kernel_shape=[1, 1], pads=[0]*4, strides=[1, 1]),
        ]
        inits = ([numpy_helper.from_array(w_pw,   f"{p}_wpw"),
                  numpy_helper.from_array(b_pw,   f"{p}_bpw"),
                  numpy_helper.from_array(w_dw,   f"{p}_wdw"),
                  numpy_helper.from_array(b_dw,   f"{p}_bdw"),
                  numpy_helper.from_array(w_proj, f"{p}_wproj"),
                  numpy_helper.from_array(b_proj, f"{p}_bproj")]
                 + bn1_inits + bn2_inits)
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 22. Conv → GELU (without BN) ─────────────────────────────────────────
class ConvGELU(OTP):
    """Direct Conv → GELU. Tests Conv-activation fusion path without BN."""
    name = "conv_gelu"
    category = CAT_FUSION
    target_optimization = "conv_gelu_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        out_c = self._rand_channels(rng)
        k = int(rng.choice([1, 3]))
        pad = k // 2
        p = self._p(node_id, "cvgelu")

        w = self._make_conv_weight(out_c, C, k)
        b = np.zeros(out_c, dtype=np.float32)
        sqrt2 = np.array([np.sqrt(2.0)], dtype=np.float32)
        half  = np.array([0.5],          dtype=np.float32)
        one   = np.array([1.0],          dtype=np.float32)

        cv_o = f"{p}_cv"; div_o = f"{p}_div"; erf_o = f"{p}_erf"
        erf1 = f"{p}_erf1"; m1 = f"{p}_m1"; out = f"{p}_out"
        nodes = [
            helper.make_node("Conv", [input_name, f"{p}_w", f"{p}_b"], [cv_o],
                             kernel_shape=[k, k], pads=[pad]*4, strides=[1, 1]),
            helper.make_node("Div",  [cv_o, f"{p}_sqrt2"], [div_o]),
            helper.make_node("Erf",  [div_o], [erf_o]),
            helper.make_node("Add",  [erf_o, f"{p}_one"], [erf1]),
            helper.make_node("Mul",  [cv_o, f"{p}_half"], [m1]),
            helper.make_node("Mul",  [m1, erf1], [out]),
        ]
        inits = [numpy_helper.from_array(w,     f"{p}_w"),
                 numpy_helper.from_array(b,     f"{p}_b"),
                 numpy_helper.from_array(sqrt2, f"{p}_sqrt2"),
                 numpy_helper.from_array(half,  f"{p}_half"),
                 numpy_helper.from_array(one,   f"{p}_one")]
        H2 = H + 2*pad - k + 1; W2 = W + 2*pad - k + 1
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, out_c, H2, W2],
                                                 ctx.dtype, ctx.layout))


# ── 23. MatMul → BiasAdd → Tanh ──────────────────────────────────────────
class MatMulBiasTanh(OTP):
    """Dense layer with Tanh. Tests GEMM+Tanh fusion path."""
    name = "matmul_bias_tanh"
    category = CAT_FUSION
    target_optimization = "gemm_tanh_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, M = ctx.shape
        K = int(rng.choice([64, 128, 256]))
        p = self._p(node_id, "mmbtanh")

        w = self._make_linear_weight(K, M)
        b = np.zeros(K, dtype=np.float32)

        mm_o = f"{p}_mm"; add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("MatMul", [input_name, f"{p}_w"], [mm_o]),
            helper.make_node("Add",    [mm_o, f"{p}_b"], [add_o]),
            helper.make_node("Tanh",   [add_o], [out]),
        ]
        inits = [numpy_helper.from_array(w, f"{p}_w"),
                 numpy_helper.from_array(b, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, K], ctx.dtype, ctx.layout))


# ── 24. AveragePool → Scale → Bias (lightweight BN substitute) ───────────
class AvgPoolScaleBias(OTP):
    """GlobalAvgPool + per-channel scale+bias. Tests pooling-normalization fusion."""
    name = "avgpool_scale_bias"
    category = CAT_FUSION
    target_optimization = "pool_norm_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "apsb")
        scale = np.ones((1, C, 1, 1), dtype=np.float32)
        bias  = np.zeros((1, C, 1, 1), dtype=np.float32)

        pool_o = f"{p}_pool"; sc_o = f"{p}_sc"; out = f"{p}_out"
        nodes = [
            helper.make_node("AveragePool", [input_name], [pool_o],
                             kernel_shape=[H, W], strides=[1, 1], pads=[0]*4),
            helper.make_node("Mul", [pool_o, f"{p}_scale"], [sc_o]),
            helper.make_node("Add", [sc_o,   f"{p}_bias"],  [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_scale"),
                 numpy_helper.from_array(bias,  f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, 1, 1], ctx.dtype, ctx.layout))


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
    # New high-value patterns
    ConvTransposeBNReLU(),
    DilatedConvBNReLU(),
    ConvBNSiLU(),
    ConvBNHardswish(),
    ConvBNReLU6(),
    GroupedConvBNReLU(),
    ConvAsymPadBN(),
    ConvBNELU(),
    PointwiseDWBlock(),
    ConvGELU(),
    MatMulBiasTanh(),
    AvgPoolScaleBias(),
]
