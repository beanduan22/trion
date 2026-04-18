"""
Attention- and transformer-sensitive OTPs.
Target: QKV fusion, softmax precision, attention masking, transformer block fusion.

All patterns operate on rank-3 tensors [B, S, D] with layout "NLC".
These are high-value for exposing TVM/TRT/ORT/XLA attention-path bugs.
"""
from __future__ import annotations
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from typing import Optional

from .base import OTP, PatternInstance, CAT_ATTENTION
from ..generation.context import StructuralContext


def _ln_params(D, prefix):
    """LayerNorm scale + bias initializers."""
    sc = np.ones(D, dtype=np.float32)
    b  = np.zeros(D, dtype=np.float32)
    return ([f"{prefix}_lnsc", f"{prefix}_lnb"],
            [numpy_helper.from_array(sc, f"{prefix}_lnsc"),
             numpy_helper.from_array(b,  f"{prefix}_lnb")])


def _linear(input_name, W, b_arr, p, tag):
    """Return (nodes, inits, out_name) for a Linear(input) = MatMul + Add."""
    wn = f"{p}_{tag}_w"; bn = f"{p}_{tag}_b"
    mm = f"{p}_{tag}_mm"; out = f"{p}_{tag}_out"
    nodes = [
        helper.make_node("MatMul", [input_name, wn], [mm]),
        helper.make_node("Add",    [mm, bn], [out]),
    ]
    inits = [numpy_helper.from_array(W, wn),
             numpy_helper.from_array(b_arr, bn)]
    return nodes, inits, out


def _gelu_nodes(input_name, p, tag):
    """Return (nodes, inits, out_name) for GELU via Erf."""
    sqrt2 = np.array([np.sqrt(2.0)], dtype=np.float32)
    half  = np.array([0.5], dtype=np.float32)
    one   = np.array([1.0], dtype=np.float32)
    div_o = f"{p}_{tag}_gdiv"; erf_o = f"{p}_{tag}_gerf"
    erf1  = f"{p}_{tag}_gerf1"; mul1  = f"{p}_{tag}_gm1"; out = f"{p}_{tag}_gout"
    nodes = [
        helper.make_node("Div", [input_name, f"{p}_{tag}_sqrt2"], [div_o]),
        helper.make_node("Erf", [div_o], [erf_o]),
        helper.make_node("Add", [erf_o, f"{p}_{tag}_one"], [erf1]),
        helper.make_node("Mul", [input_name, f"{p}_{tag}_half"], [mul1]),
        helper.make_node("Mul", [mul1, erf1], [out]),
    ]
    inits = [numpy_helper.from_array(sqrt2, f"{p}_{tag}_sqrt2"),
             numpy_helper.from_array(half,  f"{p}_{tag}_half"),
             numpy_helper.from_array(one,   f"{p}_{tag}_one")]
    return nodes, inits, out


# ── 1. Scaled Dot-Product Attention ───────────────────────────────────────
class ScaledDotProductAttention(OTP):
    """
    Q·Kᵀ / √D → Softmax → ·V
    Triggers: softmax-precision bugs, QKV matmul fusion.
    """
    name = "scaled_dot_product_attention"
    category = CAT_ATTENTION
    target_optimization = "attention_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p = self._p(node_id, "sdpa")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        kt_o = f"{p}_kt"; raw = f"{p}_raw"; sc_o = f"{p}_sc"
        sm_o = f"{p}_sm"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Transpose", [k_o], [kt_o], perm=[0, 2, 1]),
            helper.make_node("MatMul",    [q_o, kt_o], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=2),
            helper.make_node("MatMul",    [sm_o, v_o], [out]),
        ]
        inits = [
            numpy_helper.from_array(Wq,    f"{p}_wq"),
            numpy_helper.from_array(Wk,    f"{p}_wk"),
            numpy_helper.from_array(Wv,    f"{p}_wv"),
            numpy_helper.from_array(scale, f"{p}_scale"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 2. Multi-Head Self-Attention ──────────────────────────────────────────
class MultiHeadSelfAttention(OTP):
    """
    Full MHA: head-split Reshape/Transpose → per-head SDPA → merge.
    Triggers: layout-rewrite bugs, head-transpose fusion.
    """
    name = "multi_head_self_attention"
    category = CAT_ATTENTION
    target_optimization = "multi_head_attention_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 3 or ctx.dtype != "float32":
            return False
        D = ctx.shape[2]
        return D >= 32 and D % 4 == 0   # need at least 4 heads with d_h >= 8

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        H = 4 if D < 256 else 8
        if D % H != 0:
            H = 4
        dh = D // H
        p = self._p(node_id, "mhsa")

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        Wo = self._make_linear_weight(D, D)
        bq = np.zeros(D, dtype=np.float32)
        scale = np.array([1.0 / float(np.sqrt(dh))], dtype=np.float32)

        # Shape tensors for Reshape
        split_shape  = np.array([B, S, H, dh], dtype=np.int64)
        merged_shape = np.array([B, S, D],     dtype=np.int64)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        qs  = f"{p}_qs"; ks  = f"{p}_ks"; vs  = f"{p}_vs"
        qt  = f"{p}_qt"; kt  = f"{p}_kt"; vt  = f"{p}_vt"
        ktt = f"{p}_ktt"
        raw = f"{p}_raw"; sc_o = f"{p}_sc"; sm_o = f"{p}_sm"
        ctx_o = f"{p}_ctx"; ct_o = f"{p}_ct"; cm_o = f"{p}_cm"; out = f"{p}_out"

        nodes = [
            # QKV projections
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            # Split into heads [B, S, H, dh]
            helper.make_node("Reshape",   [q_o, f"{p}_sshape"], [qs]),
            helper.make_node("Reshape",   [k_o, f"{p}_sshape"], [ks]),
            helper.make_node("Reshape",   [v_o, f"{p}_sshape"], [vs]),
            # [B, H, S, dh]
            helper.make_node("Transpose", [qs], [qt], perm=[0, 2, 1, 3]),
            helper.make_node("Transpose", [ks], [kt], perm=[0, 2, 1, 3]),
            helper.make_node("Transpose", [vs], [vt], perm=[0, 2, 1, 3]),
            # Attention: K transpose [B, H, dh, S]
            helper.make_node("Transpose", [kt], [ktt], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul",    [qt, ktt], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=3),
            helper.make_node("MatMul",    [sm_o, vt], [ctx_o]),
            # Merge heads [B, S, H, dh] → [B, S, D]
            helper.make_node("Transpose", [ctx_o], [ct_o], perm=[0, 2, 1, 3]),
            helper.make_node("Reshape",   [ct_o, f"{p}_mshape"], [cm_o]),
            # Output projection
            helper.make_node("MatMul",    [cm_o, f"{p}_wo"], [out]),
        ]
        inits = [
            numpy_helper.from_array(Wq,          f"{p}_wq"),
            numpy_helper.from_array(Wk,          f"{p}_wk"),
            numpy_helper.from_array(Wv,          f"{p}_wv"),
            numpy_helper.from_array(Wo,          f"{p}_wo"),
            numpy_helper.from_array(scale,       f"{p}_scale"),
            numpy_helper.from_array(split_shape, f"{p}_sshape"),
            numpy_helper.from_array(merged_shape,f"{p}_mshape"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 3. Causal Masked Attention ────────────────────────────────────────────
class CausalMaskedAttention(OTP):
    """
    SDPA with lower-triangular mask applied via Where.
    Triggers: conditional-op handling, mask-fusion, -inf propagation bugs.
    """
    name = "causal_masked_attention"
    category = CAT_ATTENTION
    target_optimization = "causal_mask_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p = self._p(node_id, "cma")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)

        # Lower-triangular causal mask [S, S] — True = keep, False = mask out
        mask = np.tril(np.ones((S, S), dtype=bool))
        neg_inf = np.full((S, S), -1e9, dtype=np.float32)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        kt_o = f"{p}_kt"; raw = f"{p}_raw"; sc_o = f"{p}_sc"
        msk_o = f"{p}_msk"; sm_o = f"{p}_sm"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Transpose", [k_o], [kt_o], perm=[0, 2, 1]),
            helper.make_node("MatMul",    [q_o, kt_o], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            # Apply causal mask: Where(mask, score, -inf)
            helper.make_node("Where",     [f"{p}_mask", sc_o, f"{p}_neginf"], [msk_o]),
            helper.make_node("Softmax",   [msk_o], [sm_o], axis=2),
            helper.make_node("MatMul",    [sm_o, v_o], [out]),
        ]
        inits = [
            numpy_helper.from_array(Wq,     f"{p}_wq"),
            numpy_helper.from_array(Wk,     f"{p}_wk"),
            numpy_helper.from_array(Wv,     f"{p}_wv"),
            numpy_helper.from_array(scale,  f"{p}_scale"),
            numpy_helper.from_array(mask,   f"{p}_mask"),
            numpy_helper.from_array(neg_inf,f"{p}_neginf"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 4. Transformer Feed-Forward Network ───────────────────────────────────
class TransformerFFN(OTP):
    """
    Linear(4D) → GELU → Linear(D): the FFN block in transformers.
    Triggers: GEMM-GELU fusion, expansion-then-contraction pattern.
    """
    name = "transformer_ffn"
    category = CAT_ATTENTION
    target_optimization = "ffn_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        ffn_D = D * 4   # standard 4x expansion
        p = self._p(node_id, "tffn")

        W1 = self._make_linear_weight(D, ffn_D)
        b1 = np.zeros(ffn_D, dtype=np.float32)
        W2 = self._make_linear_weight(ffn_D, D)
        b2 = np.zeros(D, dtype=np.float32)

        lin1_nodes, lin1_inits, lin1_out = _linear(input_name, W1, b1, p, "l1")
        gelu_nodes, gelu_inits, gelu_out = _gelu_nodes(lin1_out, p, "gelu")
        lin2_nodes, lin2_inits, lin2_out = _linear(gelu_out, W2, b2, p, "l2")

        all_nodes = lin1_nodes + gelu_nodes + lin2_nodes
        all_inits = lin1_inits + gelu_inits + lin2_inits
        return PatternInstance(self.name, self.category, all_nodes, all_inits,
                               input_name, lin2_out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 5. Transformer Encoder Layer ──────────────────────────────────────────
class TransformerEncoderLayer(OTP):
    """
    Pre-LN → SDPA → Add → Pre-LN → FFN → Add.
    Triggers: full-block fusion, residual+norm rewrite.
    """
    name = "transformer_encoder_layer"
    category = CAT_ATTENTION
    target_optimization = "transformer_block_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32" and ctx.shape[2] >= 32

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p = self._p(node_id, "tel")
        ffn_D = min(D * 4, 2048)

        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)
        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        W1 = self._make_linear_weight(D, ffn_D)
        b1 = np.zeros(ffn_D, dtype=np.float32)
        W2 = self._make_linear_weight(ffn_D, D)
        b2 = np.zeros(D, dtype=np.float32)
        ln1_names, ln1_inits = _ln_params(D, f"{p}_ln1")
        ln2_names, ln2_inits = _ln_params(D, f"{p}_ln2")

        ln1_o = f"{p}_ln1o"; q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        kt_o  = f"{p}_kt";   raw = f"{p}_raw"; sc_o = f"{p}_sc"; sm_o = f"{p}_sm"
        att_o = f"{p}_att";  res1 = f"{p}_res1"; ln2_o = f"{p}_ln2o"
        ff1_o = f"{p}_ff1";  ff2_o = f"{p}_ff2"; out = f"{p}_out"
        sqrt2 = np.array([np.sqrt(2.0)], dtype=np.float32)
        half  = np.array([0.5],          dtype=np.float32)
        one   = np.array([1.0],          dtype=np.float32)

        nodes = [
            # Pre-LN 1
            helper.make_node("LayerNormalization",
                             [input_name] + ln1_names, [ln1_o],
                             axis=-1, epsilon=1e-5),
            # Self-attention
            helper.make_node("MatMul",    [ln1_o, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [ln1_o, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [ln1_o, f"{p}_wv"], [v_o]),
            helper.make_node("Transpose", [k_o], [kt_o], perm=[0, 2, 1]),
            helper.make_node("MatMul",    [q_o, kt_o], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=2),
            helper.make_node("MatMul",    [sm_o, v_o], [att_o]),
            # Residual 1
            helper.make_node("Add",       [input_name, att_o], [res1]),
            # Pre-LN 2
            helper.make_node("LayerNormalization",
                             [res1] + ln2_names, [ln2_o],
                             axis=-1, epsilon=1e-5),
            # FFN
            helper.make_node("MatMul",    [ln2_o, f"{p}_w1"], [f"{p}_mm1"]),
            helper.make_node("Add",       [f"{p}_mm1", f"{p}_b1"], [ff1_o]),
            helper.make_node("Div",       [ff1_o, f"{p}_sqrt2"], [f"{p}_gdiv"]),
            helper.make_node("Erf",       [f"{p}_gdiv"], [f"{p}_gerf"]),
            helper.make_node("Add",       [f"{p}_gerf", f"{p}_one"], [f"{p}_gerf1"]),
            helper.make_node("Mul",       [ff1_o, f"{p}_half"], [f"{p}_gm1"]),
            helper.make_node("Mul",       [f"{p}_gm1", f"{p}_gerf1"], [ff2_o]),
            helper.make_node("MatMul",    [ff2_o, f"{p}_w2"], [f"{p}_mm2"]),
            helper.make_node("Add",       [f"{p}_mm2", f"{p}_b2"], [f"{p}_ff_out"]),
            # Residual 2
            helper.make_node("Add",       [res1, f"{p}_ff_out"], [out]),
        ]
        inits = (ln1_inits + ln2_inits + [
            numpy_helper.from_array(scale, f"{p}_scale"),
            numpy_helper.from_array(Wq,   f"{p}_wq"),
            numpy_helper.from_array(Wk,   f"{p}_wk"),
            numpy_helper.from_array(Wv,   f"{p}_wv"),
            numpy_helper.from_array(W1,   f"{p}_w1"),
            numpy_helper.from_array(b1,   f"{p}_b1"),
            numpy_helper.from_array(W2,   f"{p}_w2"),
            numpy_helper.from_array(b2,   f"{p}_b2"),
            numpy_helper.from_array(sqrt2, f"{p}_sqrt2"),
            numpy_helper.from_array(half,  f"{p}_half"),
            numpy_helper.from_array(one,   f"{p}_one"),
        ])
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 6. Gated MLP Block (SwiGLU) ───────────────────────────────────────────
class GatedMLPBlock(OTP):
    """
    gate(x) * silu(linear(x)): SwiGLU / gated activation.
    Triggers: gate-branch fusion, sigmoid+mul patterns.
    """
    name = "gated_mlp_block"
    category = CAT_ATTENTION
    target_optimization = "swiglu_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p = self._p(node_id, "gmlp")

        W_gate = self._make_linear_weight(D, D)
        W_up   = self._make_linear_weight(D, D)
        W_down = self._make_linear_weight(D, D)

        gate_o = f"{p}_gate"; up_o = f"{p}_up"; sig_o = f"{p}_sig"
        silu_o = f"{p}_silu"; mix_o = f"{p}_mix"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",  [input_name, f"{p}_wgate"], [gate_o]),
            helper.make_node("MatMul",  [input_name, f"{p}_wup"],   [up_o]),
            # SiLU = x * sigmoid(x) applied to gate
            helper.make_node("Sigmoid", [gate_o], [sig_o]),
            helper.make_node("Mul",     [gate_o, sig_o], [silu_o]),
            # Element-wise mix
            helper.make_node("Mul",     [silu_o, up_o], [mix_o]),
            # Down projection
            helper.make_node("MatMul",  [mix_o, f"{p}_wdown"], [out]),
        ]
        inits = [
            numpy_helper.from_array(W_gate, f"{p}_wgate"),
            numpy_helper.from_array(W_up,   f"{p}_wup"),
            numpy_helper.from_array(W_down, f"{p}_wdown"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 7. Attention with Additive Position Bias ──────────────────────────────
class AttentionWithBias(OTP):
    """
    SDPA + learned [S, S] additive position bias before softmax.
    Triggers: bias-fusion, T5-style relative position encoding bugs.
    """
    name = "attention_with_bias"
    category = CAT_ATTENTION
    target_optimization = "attention_bias_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p = self._p(node_id, "awb")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        pos_bias = np.zeros((1, S, S), dtype=np.float32)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        kt  = f"{p}_kt"; raw = f"{p}_raw"; sc_o = f"{p}_sc"
        biased = f"{p}_biased"; sm_o = f"{p}_sm"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Transpose", [k_o], [kt], perm=[0, 2, 1]),
            helper.make_node("MatMul",    [q_o, kt], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Add",       [sc_o, f"{p}_pbias"], [biased]),
            helper.make_node("Softmax",   [biased], [sm_o], axis=2),
            helper.make_node("MatMul",    [sm_o, v_o], [out]),
        ]
        inits = [
            numpy_helper.from_array(Wq,       f"{p}_wq"),
            numpy_helper.from_array(Wk,       f"{p}_wk"),
            numpy_helper.from_array(Wv,       f"{p}_wv"),
            numpy_helper.from_array(scale,    f"{p}_scale"),
            numpy_helper.from_array(pos_bias, f"{p}_pbias"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 8. Group Query Attention ──────────────────────────────────────────────
class GroupQueryAttention(OTP):
    """
    Q has H heads; K/V have G < H heads, expanded with Expand.
    Triggers: Expand+MatMul fusion, GQA/MQA compiler support bugs.
    """
    name = "group_query_attention"
    category = CAT_ATTENTION
    target_optimization = "group_query_attention_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 3 or ctx.dtype != "float32":
            return False
        D = ctx.shape[2]
        return D >= 64 and D % 8 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        H  = 4           # query heads
        G  = 2           # key/value groups (G < H)
        dh = D // H
        dg = D // G      # K/V head dim = G * dh
        p  = self._p(node_id, "gqa")

        Wq = self._make_linear_weight(D, H * dh)
        Wk = self._make_linear_weight(D, G * dh)
        Wv = self._make_linear_weight(D, G * dh)
        scale = np.array([1.0 / float(np.sqrt(dh))], dtype=np.float32)

        q_shape  = np.array([B, S, H, dh],       dtype=np.int64)
        kv_shape = np.array([B, S, G, dh],       dtype=np.int64)
        # For Expand: after Transpose, K shape is [B,G,S,dh]
        # Reshape to [B,G,1,S,dh], Expand to [B,G,H//G,S,dh], Reshape to [B,H,S,dh]
        kv_unsq  = np.array([B, G, 1, S, dh],   dtype=np.int64)
        kv_exp   = np.array([B, G, H//G, S, dh], dtype=np.int64)
        kv_flat  = np.array([B, H, S, dh],       dtype=np.int64)
        merge    = np.array([B, S, H * dh],      dtype=np.int64)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        qs  = f"{p}_qs"; ks  = f"{p}_ks"; vs  = f"{p}_vs"
        qt  = f"{p}_qt"; kt  = f"{p}_kt"; vt  = f"{p}_vt"
        ku  = f"{p}_ku"; vu  = f"{p}_vu"   # unsqueezed
        ke  = f"{p}_ke"; ve  = f"{p}_ve"   # expanded
        kf  = f"{p}_kf"; vf  = f"{p}_vf"   # flattened to [B,H,S,dh]
        ktt = f"{p}_ktt"
        raw = f"{p}_raw"; sc_o = f"{p}_sc"; sm_o = f"{p}_sm"
        ctx_o = f"{p}_ctx"; ct_o = f"{p}_ct"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Reshape",   [q_o, f"{p}_qshape"],  [qs]),
            helper.make_node("Reshape",   [k_o, f"{p}_kvshape"], [ks]),
            helper.make_node("Reshape",   [v_o, f"{p}_kvshape"], [vs]),
            helper.make_node("Transpose", [qs], [qt], perm=[0, 2, 1, 3]),
            helper.make_node("Transpose", [ks], [kt], perm=[0, 2, 1, 3]),  # [B,G,S,dh]
            helper.make_node("Transpose", [vs], [vt], perm=[0, 2, 1, 3]),
            # Expand K/V: [B,G,S,dh] → [B,G,1,S,dh] → expand → [B,G,H//G,S,dh] → [B,H,S,dh]
            helper.make_node("Reshape",   [kt, f"{p}_kvunsq"], [ku]),
            helper.make_node("Expand",    [ku, f"{p}_kvexp"],   [ke]),
            helper.make_node("Reshape",   [ke, f"{p}_kvflat"],  [kf]),
            helper.make_node("Reshape",   [vt, f"{p}_kvunsq"], [vu]),
            helper.make_node("Expand",    [vu, f"{p}_kvexp"],   [ve]),
            helper.make_node("Reshape",   [ve, f"{p}_kvflat"],  [vf]),
            helper.make_node("Transpose", [kf], [ktt], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul",    [qt, ktt], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"],  [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=3),
            helper.make_node("MatMul",    [sm_o, vf], [ctx_o]),
            helper.make_node("Transpose", [ctx_o], [ct_o], perm=[0, 2, 1, 3]),
            helper.make_node("Reshape",   [ct_o, f"{p}_merge"], [out]),
        ]
        inits = [
            numpy_helper.from_array(Wq,      f"{p}_wq"),
            numpy_helper.from_array(Wk,      f"{p}_wk"),
            numpy_helper.from_array(Wv,      f"{p}_wv"),
            numpy_helper.from_array(scale,   f"{p}_scale"),
            numpy_helper.from_array(q_shape, f"{p}_qshape"),
            numpy_helper.from_array(kv_shape,f"{p}_kvshape"),
            numpy_helper.from_array(kv_unsq, f"{p}_kvunsq"),
            numpy_helper.from_array(kv_exp,  f"{p}_kvexp"),
            numpy_helper.from_array(kv_flat, f"{p}_kvflat"),
            numpy_helper.from_array(merge,   f"{p}_merge"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, H * dh], ctx.dtype, "NLC"))


# ── 9. KV-Cache Concat Attention ──────────────────────────────────────────
class KVCacheAttention(OTP):
    """
    Concat past_K/past_V then SDPA: simulates KV-cache extension.
    Triggers: Concat shape-inference bugs, variable-length attention.
    """
    name = "kv_cache_attention"
    category = CAT_ATTENTION
    target_optimization = "kv_cache_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        past_S = max(1, S // 2)   # half the sequence as "cached" past
        total_S = S + past_S
        p = self._p(node_id, "kvca")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        past_K = np.zeros((B, past_S, D), dtype=np.float32)
        past_V = np.zeros((B, past_S, D), dtype=np.float32)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        k_ext = f"{p}_kext"; v_ext = f"{p}_vext"
        kt    = f"{p}_kt";   raw  = f"{p}_raw"
        sc_o  = f"{p}_sc";   sm_o = f"{p}_sm"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            # Extend K/V with cached past
            helper.make_node("Concat",    [f"{p}_pastk", k_o], [k_ext], axis=1),
            helper.make_node("Concat",    [f"{p}_pastv", v_o], [v_ext], axis=1),
            helper.make_node("Transpose", [k_ext], [kt], perm=[0, 2, 1]),
            helper.make_node("MatMul",    [q_o, kt], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=2),
            helper.make_node("MatMul",    [sm_o, v_ext], [out]),
        ]
        inits = [
            numpy_helper.from_array(Wq,     f"{p}_wq"),
            numpy_helper.from_array(Wk,     f"{p}_wk"),
            numpy_helper.from_array(Wv,     f"{p}_wv"),
            numpy_helper.from_array(scale,  f"{p}_scale"),
            numpy_helper.from_array(past_K, f"{p}_pastk"),
            numpy_helper.from_array(past_V, f"{p}_pastv"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 10. Rotary Position Embedding Attention ───────────────────────────────
class RotaryEmbeddingAttention(OTP):
    """
    Apply RoPE-style cos/sin rotation to Q/K before SDPA.
    Triggers: cos/sin const-fold, element-wise rotation fusion bugs.
    """
    name = "rotary_embedding_attention"
    category = CAT_ATTENTION
    target_optimization = "rope_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 3 or ctx.dtype != "float32":
            return False
        return ctx.shape[2] >= 16 and ctx.shape[2] % 2 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        half = D // 2
        p = self._p(node_id, "rope")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)

        # Precomputed cos/sin for positions 0..S-1 on half dimensions
        pos    = np.arange(S, dtype=np.float32)
        freq   = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
        theta  = np.outer(pos, freq)           # [S, half]
        cos_   = np.cos(theta).astype(np.float32)
        sin_   = np.sin(theta).astype(np.float32)
        # Broadcast shapes: [1, S, half]
        cos_b  = cos_[np.newaxis, :, :]
        sin_b  = sin_[np.newaxis, :, :]
        neg1   = np.array([-1.0], dtype=np.float32)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        # Slice axis 2 (D dimension) only — all arrays must be 1-element to match ax
        sl1  = np.array([0],    dtype=np.int64)   # start of first half
        el1  = np.array([half], dtype=np.int64)   # end of first half
        sl2  = np.array([half], dtype=np.int64)   # start of second half
        el2  = np.array([D],    dtype=np.int64)   # end of second half
        step = np.array([1],    dtype=np.int64)   # step = 1

        q1 = f"{p}_q1"; q2 = f"{p}_q2"
        q_rot1 = f"{p}_qr1"; q_rot2 = f"{p}_qr2"; q_rot = f"{p}_qrot"
        k1 = f"{p}_k1"; k2 = f"{p}_k2"
        k_rot1 = f"{p}_kr1"; k_rot2 = f"{p}_kr2"; k_rot = f"{p}_krot"
        kt = f"{p}_kt"; raw = f"{p}_raw"; sc_o = f"{p}_sc"
        sm_o = f"{p}_sm"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul", [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul", [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul", [input_name, f"{p}_wv"], [v_o]),
            # Slice Q into two halves
            helper.make_node("Slice", [q_o, f"{p}_sl1", f"{p}_el1", f"{p}_ax", f"{p}_step"], [q1]),
            helper.make_node("Slice", [q_o, f"{p}_sl2", f"{p}_el2", f"{p}_ax", f"{p}_step"], [q2]),
            # RoPE on Q: q_rot = [q1*cos - q2*sin, q2*cos + q1*sin]
            helper.make_node("Mul", [q1, f"{p}_cos"], [f"{p}_q1cos"]),
            helper.make_node("Mul", [q2, f"{p}_sin"], [f"{p}_q2sin"]),
            helper.make_node("Mul", [f"{p}_q2sin", f"{p}_neg1"], [f"{p}_q2sinn"]),
            helper.make_node("Add", [f"{p}_q1cos", f"{p}_q2sinn"], [q_rot1]),
            helper.make_node("Mul", [q2, f"{p}_cos"], [f"{p}_q2cos"]),
            helper.make_node("Mul", [q1, f"{p}_sin"], [f"{p}_q1sin"]),
            helper.make_node("Add", [f"{p}_q2cos", f"{p}_q1sin"], [q_rot2]),
            helper.make_node("Concat", [q_rot1, q_rot2], [q_rot], axis=2),
            # Slice K into two halves
            helper.make_node("Slice", [k_o, f"{p}_sl1", f"{p}_el1", f"{p}_ax", f"{p}_step"], [k1]),
            helper.make_node("Slice", [k_o, f"{p}_sl2", f"{p}_el2", f"{p}_ax", f"{p}_step"], [k2]),
            helper.make_node("Mul", [k1, f"{p}_cos"], [f"{p}_k1cos"]),
            helper.make_node("Mul", [k2, f"{p}_sin"], [f"{p}_k2sin"]),
            helper.make_node("Mul", [f"{p}_k2sin", f"{p}_neg1"], [f"{p}_k2sinn"]),
            helper.make_node("Add", [f"{p}_k1cos", f"{p}_k2sinn"], [k_rot1]),
            helper.make_node("Mul", [k2, f"{p}_cos"], [f"{p}_k2cos"]),
            helper.make_node("Mul", [k1, f"{p}_sin"], [f"{p}_k1sin"]),
            helper.make_node("Add", [f"{p}_k2cos", f"{p}_k1sin"], [k_rot2]),
            helper.make_node("Concat", [k_rot1, k_rot2], [k_rot], axis=2),
            # SDPA with rotated Q/K
            helper.make_node("Transpose", [k_rot], [kt], perm=[0, 2, 1]),
            helper.make_node("MatMul",    [q_rot, kt], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=2),
            helper.make_node("MatMul",    [sm_o, v_o], [out]),
        ]
        ax = np.array([2], dtype=np.int64)
        inits = [
            numpy_helper.from_array(Wq,    f"{p}_wq"),
            numpy_helper.from_array(Wk,    f"{p}_wk"),
            numpy_helper.from_array(Wv,    f"{p}_wv"),
            numpy_helper.from_array(scale, f"{p}_scale"),
            numpy_helper.from_array(cos_b, f"{p}_cos"),
            numpy_helper.from_array(sin_b, f"{p}_sin"),
            numpy_helper.from_array(neg1,  f"{p}_neg1"),
            numpy_helper.from_array(sl1,   f"{p}_sl1"),
            numpy_helper.from_array(el1,   f"{p}_el1"),
            numpy_helper.from_array(sl2,   f"{p}_sl2"),
            numpy_helper.from_array(el2,   f"{p}_el2"),
            numpy_helper.from_array(ax,    f"{p}_ax"),
            numpy_helper.from_array(step,  f"{p}_step"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 11. Self-Attention with Residual & Layer Norm ─────────────────────────
class SelfAttentionResidual(OTP):
    """
    SDPA + residual connection + post-LN.
    Triggers: residual+norm fusion (common in both TVM and ORT).
    """
    name = "self_attention_residual"
    category = CAT_ATTENTION
    target_optimization = "attn_residual_norm_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p = self._p(node_id, "sar")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        ln_names, ln_inits = _ln_params(D, p)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        kt  = f"{p}_kt"; raw = f"{p}_raw"; sc_o = f"{p}_sc"
        sm_o = f"{p}_sm"; att_o = f"{p}_att"
        res  = f"{p}_res"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",              [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",              [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",              [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Transpose",           [k_o], [kt], perm=[0, 2, 1]),
            helper.make_node("MatMul",              [q_o, kt], [raw]),
            helper.make_node("Mul",                 [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",             [sc_o], [sm_o], axis=2),
            helper.make_node("MatMul",              [sm_o, v_o], [att_o]),
            helper.make_node("Add",                 [input_name, att_o], [res]),
            helper.make_node("LayerNormalization",  [res] + ln_names, [out],
                             axis=-1, epsilon=1e-5),
        ]
        inits = ln_inits + [
            numpy_helper.from_array(Wq,    f"{p}_wq"),
            numpy_helper.from_array(Wk,    f"{p}_wk"),
            numpy_helper.from_array(Wv,    f"{p}_wv"),
            numpy_helper.from_array(scale, f"{p}_scale"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 12. Multi-Query Attention ─────────────────────────────────────────────
class MultiQueryAttention(OTP):
    """
    Multiple Q heads sharing a single K/V (MQA).
    Triggers: broadcast in attention, key sharing fusion bugs.
    """
    name = "multi_query_attention"
    category = CAT_ATTENTION
    target_optimization = "multi_query_attention_fusion"

    def is_compatible(self, ctx):
        if ctx.rank != 3 or ctx.dtype != "float32":
            return False
        D = ctx.shape[2]
        return D >= 32 and D % 4 == 0

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        H  = 4
        dh = D // H
        p  = self._p(node_id, "mqa")
        scale = np.array([1.0 / float(np.sqrt(dh))], dtype=np.float32)

        # H Q projections, but only 1 K/V projection
        Wq = self._make_linear_weight(D, H * dh)
        Wk = self._make_linear_weight(D, dh)
        Wv = self._make_linear_weight(D, dh)
        Wo = self._make_linear_weight(H * dh, D)

        q_shape   = np.array([B, S, H, dh], dtype=np.int64)
        out_shape = np.array([B, S, H * dh], dtype=np.int64)
        # K/V need to be broadcast: [B, 1, S, dh] → expand to [B, H, S, dh]
        kv_us  = np.array([B, 1, S, dh],    dtype=np.int64)
        kv_exp = np.array([B, H, S, dh],    dtype=np.int64)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        qs  = f"{p}_qs"; qt  = f"{p}_qt"
        kus = f"{p}_kus"; vus = f"{p}_vus"
        ke  = f"{p}_ke";  ve  = f"{p}_ve"
        ktt = f"{p}_ktt"
        raw = f"{p}_raw"; sc_o = f"{p}_sc"; sm_o = f"{p}_sm"
        ctx_o = f"{p}_ctx"; ct_o = f"{p}_ct"; cm_o = f"{p}_cm"; out = f"{p}_out"

        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Reshape",   [q_o, f"{p}_qshape"], [qs]),
            helper.make_node("Transpose", [qs], [qt], perm=[0, 2, 1, 3]),
            # Unsqueeze K/V to add head dim, then expand
            helper.make_node("Reshape",   [k_o, f"{p}_kvus"], [kus]),
            helper.make_node("Expand",    [kus, f"{p}_kvexp"], [ke]),
            helper.make_node("Reshape",   [v_o, f"{p}_kvus"], [vus]),
            helper.make_node("Expand",    [vus, f"{p}_kvexp"], [ve]),
            helper.make_node("Transpose", [ke], [ktt], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul",    [qt, ktt], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=3),
            helper.make_node("MatMul",    [sm_o, ve], [ctx_o]),
            helper.make_node("Transpose", [ctx_o], [ct_o], perm=[0, 2, 1, 3]),
            helper.make_node("Reshape",   [ct_o, f"{p}_oshape"], [cm_o]),
            helper.make_node("MatMul",    [cm_o, f"{p}_wo"], [out]),
        ]
        inits = [
            numpy_helper.from_array(Wq,       f"{p}_wq"),
            numpy_helper.from_array(Wk,       f"{p}_wk"),
            numpy_helper.from_array(Wv,       f"{p}_wv"),
            numpy_helper.from_array(Wo,       f"{p}_wo"),
            numpy_helper.from_array(scale,    f"{p}_scale"),
            numpy_helper.from_array(q_shape,  f"{p}_qshape"),
            numpy_helper.from_array(kv_us,    f"{p}_kvus"),
            numpy_helper.from_array(kv_exp,   f"{p}_kvexp"),
            numpy_helper.from_array(out_shape, f"{p}_oshape"),
        ]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 13. Attention with additive relative position bias ────────────────────
class RelativePositionBiasAttention(OTP):
    """SDPA + learned relative position bias matrix added before softmax.
    Tests attention rewriting when a non-trivial additive bias is present."""
    name = "relative_position_bias_attention"
    category = CAT_ATTENTION
    target_optimization = "attention_with_position_bias_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p     = self._p(node_id, "rpba")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)
        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        # Relative position bias: [1, S, S] broadcastable to [B, S, S]
        pos_bias = np.zeros((1, S, S), dtype=np.float32)

        q_o = f"{p}_q"; k_o = f"{p}_k"; v_o = f"{p}_v"
        kt_o = f"{p}_kt"; raw = f"{p}_raw"; sc_o = f"{p}_sc"
        pb_o = f"{p}_pb"; sm_o = f"{p}_sm"; out = f"{p}_out"
        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Transpose", [k_o], [kt_o], perm=[0, 2, 1]),
            helper.make_node("MatMul",    [q_o, kt_o], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Add",       [sc_o, f"{p}_pos_bias"], [pb_o]),
            helper.make_node("Softmax",   [pb_o], [sm_o], axis=2),
            helper.make_node("MatMul",    [sm_o, v_o], [out]),
        ]
        inits = [numpy_helper.from_array(Wq,      f"{p}_wq"),
                 numpy_helper.from_array(Wk,      f"{p}_wk"),
                 numpy_helper.from_array(Wv,      f"{p}_wv"),
                 numpy_helper.from_array(scale,   f"{p}_scale"),
                 numpy_helper.from_array(pos_bias, f"{p}_pos_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 14. SwiGLU FFN (LLaMA-style): gate * silu(gate) + up_proj ────────────
class SwiGLUFFN(OTP):
    """SwiGLU: gate_proj → SiLU, up_proj → multiply, then down_proj.
    Critical transformer block in LLaMA/Mistral; stresses gate fusion."""
    name = "swiglu_ffn"
    category = CAT_ATTENTION
    target_optimization = "swiglu_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        ff = D * 4
        p  = self._p(node_id, "swgl")
        Wg = self._make_linear_weight(D, ff)
        Wu = self._make_linear_weight(D, ff)
        Wd = self._make_linear_weight(ff, D)
        bg = np.zeros(ff, dtype=np.float32)
        bu = np.zeros(ff, dtype=np.float32)
        bd = np.zeros(D,  dtype=np.float32)

        g_o = f"{p}_g"; u_o = f"{p}_u"
        sig_o = f"{p}_sig"; silu_o = f"{p}_silu"
        h_o = f"{p}_h"; mm_o = f"{p}_mm"; out = f"{p}_out"
        nodes = [
            helper.make_node("MatMul",  [input_name, f"{p}_wg"], [g_o]),
            helper.make_node("Add",     [g_o, f"{p}_bg"], [f"{p}_ga"]),
            helper.make_node("MatMul",  [input_name, f"{p}_wu"], [u_o]),
            helper.make_node("Add",     [u_o, f"{p}_bu"], [f"{p}_ua"]),
            # SiLU on gate: x * sigmoid(x)
            helper.make_node("Sigmoid", [f"{p}_ga"], [sig_o]),
            helper.make_node("Mul",     [f"{p}_ga", sig_o], [silu_o]),
            # Element-wise multiply gated activation with up path
            helper.make_node("Mul",     [silu_o, f"{p}_ua"], [h_o]),
            helper.make_node("MatMul",  [h_o, f"{p}_wd"], [mm_o]),
            helper.make_node("Add",     [mm_o, f"{p}_bd"], [out]),
        ]
        inits = [numpy_helper.from_array(Wg, f"{p}_wg"),
                 numpy_helper.from_array(Wu, f"{p}_wu"),
                 numpy_helper.from_array(Wd, f"{p}_wd"),
                 numpy_helper.from_array(bg, f"{p}_bg"),
                 numpy_helper.from_array(bu, f"{p}_bu"),
                 numpy_helper.from_array(bd, f"{p}_bd")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B, S, D], ctx.dtype, "NLC"))


# ── 15. Cross-Attention (encoder-decoder) ─────────────────────────────────
class CrossAttention(OTP):
    """Decoder cross-attention: Q from input, KV from fixed memory tensor."""
    name = "cross_attention"
    category = CAT_ATTENTION
    target_optimization = "cross_attention_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32" and ctx.shape[2] >= 32

    def instantiate(self, input_name, ctx, rng, node_id):
        B, T, D = ctx.shape   # target sequence
        S = T * 2             # encoder memory length
        p = self._p(node_id, "ca")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        Wo = self._make_linear_weight(D, D)
        # Fixed encoder memory
        memory = np.random.default_rng(node_id).standard_normal((B, S, D)).astype(np.float32) * 0.02

        q_o=f"{p}_q"; k_o=f"{p}_k"; v_o=f"{p}_v"
        kt=f"{p}_kt"
        raw=f"{p}_raw"; sc_o=f"{p}_sc"; sm_o=f"{p}_sm"
        att=f"{p}_att"; out=f"{p}_out"
        # Q: [B,T,D]; K,V: [B,S,D] (from memory).
        # Scores = Q @ K^T → [B,T,S]; Context = softmax(scores) @ V → [B,T,D].
        nodes = [
            helper.make_node("MatMul", [input_name,     f"{p}_wq"], [q_o]),
            helper.make_node("MatMul", [f"{p}_mem",     f"{p}_wk"], [k_o]),
            helper.make_node("MatMul", [f"{p}_mem",     f"{p}_wv"], [v_o]),
            helper.make_node("Transpose", [k_o], [kt], perm=[0,2,1]),   # [B,D,S]
            helper.make_node("MatMul",    [q_o, kt], [raw]),            # [B,T,S]
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Softmax",   [sc_o], [sm_o], axis=-1),
            helper.make_node("MatMul",    [sm_o, v_o], [att]),          # [B,T,D]
            helper.make_node("MatMul",    [att, f"{p}_wo"], [out]),
        ]
        inits = [numpy_helper.from_array(Wq,     f"{p}_wq"),
                 numpy_helper.from_array(Wk,     f"{p}_wk"),
                 numpy_helper.from_array(Wv,     f"{p}_wv"),
                 numpy_helper.from_array(Wo,     f"{p}_wo"),
                 numpy_helper.from_array(memory, f"{p}_mem"),
                 numpy_helper.from_array(scale,  f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B,T,D], ctx.dtype, ctx.layout))


# ── 16. ALiBi Attention (linear position bias, no learned embedding) ───────
class ALiBiAttention(OTP):
    """SDPA with additive ALiBi position bias: head-specific slope * distance."""
    name = "alibi_attention"
    category = CAT_ATTENTION
    target_optimization = "alibi_position_bias_fusion"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32" and ctx.shape[2] >= 32

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        p = self._p(node_id, "alibi")
        scale = np.array([1.0 / float(np.sqrt(D))], dtype=np.float32)

        Wq = self._make_linear_weight(D, D)
        Wk = self._make_linear_weight(D, D)
        Wv = self._make_linear_weight(D, D)
        # ALiBi bias: [1, S, S] — slope * (j - i) for causal mask
        positions = np.arange(S, dtype=np.float32)
        bias = (positions[None, :] - positions[:, None]) * (-0.125)
        bias = np.clip(bias, -1e4, 0).astype(np.float32)[None]  # [1,S,S]

        q_o=f"{p}_q"; k_o=f"{p}_k"; v_o=f"{p}_v"
        qt=f"{p}_qt"; raw=f"{p}_raw"; sc_o=f"{p}_sc"
        biased=f"{p}_biased"; sm_o=f"{p}_sm"; out=f"{p}_out"
        nodes = [
            helper.make_node("MatMul",    [input_name, f"{p}_wq"], [q_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wk"], [k_o]),
            helper.make_node("MatMul",    [input_name, f"{p}_wv"], [v_o]),
            helper.make_node("Transpose", [k_o], [qt], perm=[0,2,1]),
            helper.make_node("MatMul",    [q_o, qt], [raw]),
            helper.make_node("Mul",       [raw, f"{p}_scale"], [sc_o]),
            helper.make_node("Add",       [sc_o, f"{p}_bias"], [biased]),
            helper.make_node("Softmax",   [biased], [sm_o], axis=-1),
            helper.make_node("MatMul",    [sm_o, v_o], [out]),
        ]
        inits = [numpy_helper.from_array(Wq,   f"{p}_wq"),
                 numpy_helper.from_array(Wk,   f"{p}_wk"),
                 numpy_helper.from_array(Wv,   f"{p}_wv"),
                 numpy_helper.from_array(scale, f"{p}_scale"),
                 numpy_helper.from_array(bias,  f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(3, [B,S,D], ctx.dtype, ctx.layout))


# ── 17. FFN with Dropout (training pattern) ───────────────────────────────
class FFNWithDropout(OTP):
    """Linear → GELU → Dropout(p=0) → Linear: training-graph residual FFN."""
    name = "ffn_with_dropout"
    category = CAT_ATTENTION
    target_optimization = "ffn_dropout_elim"

    def is_compatible(self, ctx):
        return ctx.rank == 3 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        B, S, D = ctx.shape
        ff = D * 4
        p  = self._p(node_id, "ffnd")
        W1 = self._make_linear_weight(D, ff)
        b1 = np.zeros(ff, dtype=np.float32)
        W2 = self._make_linear_weight(ff, D)
        b2 = np.zeros(D, dtype=np.float32)
        sqrt2 = np.array([np.sqrt(2.0)], dtype=np.float32)
        half  = np.array([0.5], dtype=np.float32)
        one   = np.array([1.0], dtype=np.float32)

        lin1_nodes, lin1_inits, lin1_out = _linear(input_name, W1, b1, p, "l1")
        gelu_nodes, gelu_inits, gelu_out = _gelu_nodes(lin1_out, p, "ge")
        # Dropout with ratio=0 (identity at inference, but graph has the node)
        drop_out = f"{p}_drop"
        ratio = np.array(0.0, dtype=np.float32)
        drop_nodes = [helper.make_node("Dropout", [gelu_out, f"{p}_ratio"], [drop_out])]
        drop_inits = [numpy_helper.from_array(ratio, f"{p}_ratio")]
        lin2_nodes, lin2_inits, lin2_out = _linear(drop_out, W2, b2, p, "l2")

        all_nodes = lin1_nodes + gelu_nodes + drop_nodes + lin2_nodes
        all_inits = lin1_inits + gelu_inits + drop_inits + lin2_inits
        return PatternInstance(self.name, self.category, all_nodes, all_inits,
                               input_name, lin2_out,
                               StructuralContext(3, [B,S,D], ctx.dtype, ctx.layout))


ALL_ATTENTION_PATTERNS = [
    ScaledDotProductAttention(),
    MultiHeadSelfAttention(),
    CausalMaskedAttention(),
    TransformerFFN(),
    TransformerEncoderLayer(),
    GatedMLPBlock(),
    AttentionWithBias(),
    GroupQueryAttention(),
    KVCacheAttention(),
    RotaryEmbeddingAttention(),
    SelfAttentionResidual(),
    MultiQueryAttention(),
    RelativePositionBiasAttention(),
    SwiGLUFFN(),
    CrossAttention(),
    ALiBiAttention(),
    FFNWithDropout(),
]
