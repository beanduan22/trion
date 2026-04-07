"""
Constant- and canonicalization-sensitive OTPs.
Target: constant folding, canonicalization, redundant-op elimination.
"""
from __future__ import annotations
import numpy as np
from onnx import helper, numpy_helper
from typing import Optional

from .base import OTP, PatternInstance, CAT_CONSTANT
from ..generation.context import StructuralContext


# ── 1. Constant → Add → Mul (const-folding trigger) ──────────────────────
class ConstantAddMul(OTP):
    name = "constant_add_mul"
    category = CAT_CONSTANT
    target_optimization = "constant_folding"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "cam")
        c1 = np.full(ctx.shape, 0.1, dtype=np.float32)
        c2 = np.full(ctx.shape, 1.01, dtype=np.float32)

        add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Add", [input_name, f"{p}_c1"], [add_o]),
            helper.make_node("Mul", [add_o,      f"{p}_c2"], [out]),
        ]
        inits = [numpy_helper.from_array(c1, f"{p}_c1"),
                 numpy_helper.from_array(c2, f"{p}_c2")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 2. Mul(1) → Add(0) (identity chain — should be folded away) ───────────
class IdentityChain(OTP):
    name = "identity_chain"
    category = CAT_CONSTANT
    target_optimization = "identity_elimination"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "idc")
        ones  = np.ones(ctx.shape,  dtype=np.float32)
        zeros = np.zeros(ctx.shape, dtype=np.float32)

        mul_o = f"{p}_mul"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul", [input_name,  f"{p}_ones"],  [mul_o]),
            helper.make_node("Add", [mul_o,        f"{p}_zeros"], [out]),
        ]
        inits = [numpy_helper.from_array(ones,  f"{p}_ones"),
                 numpy_helper.from_array(zeros, f"{p}_zeros")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 3. Reshape → Flatten → Reshape (redundant reshape) ────────────────────
class RedundantReshape(OTP):
    name = "redundant_reshape"
    category = CAT_CONSTANT
    target_optimization = "redundant_reshape_elimination"

    def is_compatible(self, ctx):
        return ctx.rank >= 2

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "rrsh")
        N = ctx.shape[0]
        flat = ctx.num_elements() // N
        s_flat = np.array([-1, flat], dtype=np.int64)
        s_orig = np.array([-1] + list(ctx.shape[1:]), dtype=np.int64)

        r1_o = f"{p}_r1"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_sfl"], [r1_o]),
            helper.make_node("Reshape", [r1_o, f"{p}_sor"], [out]),
        ]
        inits = [numpy_helper.from_array(s_flat, f"{p}_sfl"),
                 numpy_helper.from_array(s_orig, f"{p}_sor")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 4. Sub(x,x) (zero-out — constant-foldable) ────────────────────────────
class SelfSubZero(OTP):
    name = "self_sub_zero"
    category = CAT_CONSTANT
    target_optimization = "self_subtraction_constant_fold"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "ssz")
        zero = np.zeros(ctx.shape, dtype=np.float32)

        sub_o = f"{p}_sub"; out = f"{p}_out"
        # sub(x, x) = 0 then add back a learnable bias
        bias = np.zeros(ctx.shape, dtype=np.float32)
        nodes = [
            helper.make_node("Sub", [input_name, input_name], [sub_o]),
            helper.make_node("Add", [input_name, f"{p}_bias"], [out]),   # actual path
        ]
        inits = [numpy_helper.from_array(bias, f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 5. Div(x, Constant) → Mul(1/c) rewrite ────────────────────────────────
class DivByConstant(OTP):
    name = "div_by_constant"
    category = CAT_CONSTANT
    target_optimization = "div_const_to_mul_rewrite"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "dbc")
        c = np.array([5.25], dtype=np.float32)
        inv_c = np.array([1.0 / float(c)], dtype=np.float32)

        out = f"{p}_out"
        # Both representations of the same op — compiler should treat them equal
        nodes = [
            helper.make_node("Div", [input_name, f"{p}_c"], [f"{p}_div"]),
            helper.make_node("Mul", [input_name, f"{p}_ic"], [f"{p}_mul"]),
            helper.make_node("Add", [f"{p}_div",  f"{p}_mul"], [f"{p}_sum"]),
            # Average the two equivalent paths
            helper.make_node("Div", [f"{p}_sum",  f"{p}_two"], [out]),
        ]
        two = np.array([2.0], dtype=np.float32)
        inits = [numpy_helper.from_array(c,    f"{p}_c"),
                 numpy_helper.from_array(inv_c, f"{p}_ic"),
                 numpy_helper.from_array(two,   f"{p}_two")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 6. Pow(x,1) / Pow(x,2) (power canonicalization) ─────────────────────
class PowCanonical(OTP):
    name = "pow_canonical"
    category = CAT_CONSTANT
    target_optimization = "pow_constant_folding"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "pwc")
        two  = np.array([2.0], dtype=np.float32)
        eps  = np.array([1e-6], dtype=np.float32)
        abs_o = f"{p}_abs"; add_o = f"{p}_add"
        sq_o  = f"{p}_sq";  out   = f"{p}_out"
        nodes = [
            helper.make_node("Abs",  [input_name], [abs_o]),
            helper.make_node("Add",  [abs_o, f"{p}_eps"], [add_o]),
            helper.make_node("Pow",  [add_o, f"{p}_two"], [sq_o]),
            helper.make_node("Sqrt", [sq_o], [out]),    # sqrt(x^2) ≈ |x|
        ]
        inits = [numpy_helper.from_array(two, f"{p}_two"),
                 numpy_helper.from_array(eps, f"{p}_eps")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 7. Cast → Cast (round-trip dtype) ─────────────────────────────────────
class CastRoundTrip(OTP):
    name = "cast_roundtrip"
    category = CAT_CONSTANT
    target_optimization = "cast_elimination"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        import onnx
        p = self._p(node_id, "crt")
        # float32 → float16 → float32 (precision loss test)
        f16_o = f"{p}_f16"; out = f"{p}_out"
        nodes = [
            helper.make_node("Cast", [input_name], [f16_o],
                             to=onnx.TensorProto.FLOAT16),
            helper.make_node("Cast", [f16_o], [out],
                             to=onnx.TensorProto.FLOAT),
        ]
        return PatternInstance(self.name, self.category, nodes, [],
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 8. Where with constant condition (should be folded) ────────────────────
class WhereConstCond(OTP):
    name = "where_const_cond"
    category = CAT_CONSTANT
    target_optimization = "where_constant_condition_fold"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "wcc")
        # Always-true condition → should select x branch
        cond  = np.ones(ctx.shape, dtype=bool)
        other = np.zeros(ctx.shape, dtype=np.float32)

        out = f"{p}_out"
        nodes = [
            helper.make_node("Where", [f"{p}_cond", input_name, f"{p}_other"], [out]),
        ]
        inits = [numpy_helper.from_array(cond,  f"{p}_cond"),
                 numpy_helper.from_array(other, f"{p}_other")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 9. Sqrt → Reciprocal → Mul (RSqrt pattern) ───────────────────────────
class SqrtReciprocalMul(OTP):
    """Sqrt → Reciprocal → Mul: should simplify to RSqrt+Mul or fused norm."""
    name = "sqrt_reciprocal_mul"
    category = CAT_CONSTANT
    target_optimization = "rsqrt_fusion"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "srm")
        eps  = np.array([1e-6], dtype=np.float32)
        scale = np.ones(ctx.shape, dtype=np.float32)

        sq2_o = f"{p}_sq2"; add_o = f"{p}_add"; sq_o = f"{p}_sq"
        rc_o  = f"{p}_rc";  out   = f"{p}_out"
        nodes = [
            # x^2 + eps is always positive, so Sqrt and Reciprocal are safe
            helper.make_node("Mul",        [input_name, input_name], [sq2_o]),
            helper.make_node("Add",        [sq2_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt",       [add_o], [sq_o]),
            helper.make_node("Reciprocal", [sq_o], [rc_o]),
            helper.make_node("Mul",        [input_name, rc_o], [f"{p}_norm"]),
            helper.make_node("Mul",        [f"{p}_norm", f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(eps,   f"{p}_eps"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 10. Transpose → Transpose (identity pair) ─────────────────────────────
class TransposeInverseCancel(OTP):
    """T(perm) → T(perm^-1): should reduce to identity by compiler."""
    name = "transpose_inverse_cancel"
    category = CAT_CONSTANT
    target_optimization = "transpose_elimination"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "tic")
        perm     = [0, 2, 3, 1]   # NCHW → NHWC
        inv_perm = [0, 3, 1, 2]   # NHWC → NCHW

        tr1_o = f"{p}_tr1"; out = f"{p}_out"
        nodes = [
            helper.make_node("Transpose", [input_name], [tr1_o], perm=perm),
            helper.make_node("Transpose", [tr1_o], [out], perm=inv_perm),
        ]
        return PatternInstance(self.name, self.category, nodes, [],
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 11. Reshape → Reshape cancel pair ─────────────────────────────────────
class ReshapeReshapeCancel(OTP):
    """x → flatten → reshape back: compiler should fold to identity."""
    name = "reshape_reshape_cancel"
    category = CAT_CONSTANT
    target_optimization = "reshape_elimination"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "rrc")
        flat = np.array([N, C * H * W], dtype=np.int64)
        back = np.array([N, C, H, W],   dtype=np.int64)

        r1_o = f"{p}_r1"; out = f"{p}_out"
        nodes = [
            helper.make_node("Reshape", [input_name, f"{p}_flat"], [r1_o]),
            helper.make_node("Reshape", [r1_o, f"{p}_back"], [out]),
        ]
        inits = [numpy_helper.from_array(flat, f"{p}_flat"),
                 numpy_helper.from_array(back, f"{p}_back")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 12. Slice full range (identity slice) ────────────────────────────────
class SliceFullRange(OTP):
    """Slice that covers entire tensor: should be removed as no-op."""
    name = "slice_full_range"
    category = CAT_CONSTANT
    target_optimization = "slice_noop_elimination"

    def is_compatible(self, ctx):
        return ctx.rank >= 2 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "sfr")
        starts = np.zeros(ctx.rank, dtype=np.int64)
        ends   = np.array([s + 1000000 for s in ctx.shape], dtype=np.int64)
        scale  = np.ones(ctx.shape, dtype=np.float32)

        sl_o = f"{p}_sl"; out = f"{p}_out"
        nodes = [
            helper.make_node("Slice", [input_name, f"{p}_start", f"{p}_end"], [sl_o]),
            helper.make_node("Mul",   [sl_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(starts, f"{p}_start"),
                 numpy_helper.from_array(ends,   f"{p}_end"),
                 numpy_helper.from_array(scale,  f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 13. Zero Pad + Slice back (no-op round trip) ─────────────────────────
class PadSliceNoop(OTP):
    """Pad(zeros) then Slice back: identical to identity, tests round-trip opt."""
    name = "pad_slice_noop"
    category = CAT_CONSTANT
    target_optimization = "pad_slice_roundtrip_elimination"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "psn")
        pad_size = 2
        pads_t = np.array([0,0,pad_size,pad_size, 0,0,pad_size,pad_size],
                          dtype=np.int64)
        starts = np.array([0, 0, pad_size, pad_size], dtype=np.int64)
        ends   = np.array([N, C, H + pad_size, W + pad_size], dtype=np.int64)

        pd_o = f"{p}_pd"; out = f"{p}_out"
        nodes = [
            helper.make_node("Pad",   [input_name, f"{p}_pads"], [pd_o], mode="constant"),
            helper.make_node("Slice", [pd_o, f"{p}_start", f"{p}_end"], [out]),
        ]
        inits = [numpy_helper.from_array(pads_t, f"{p}_pads"),
                 numpy_helper.from_array(starts, f"{p}_start"),
                 numpy_helper.from_array(ends,   f"{p}_end")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, H, W], ctx.dtype, ctx.layout))


# ── 14. Mul by Reciprocal of constant ─────────────────────────────────────
class MulByReciprocal(OTP):
    """x * (1/c): should fold to x / c. Tests const reciprocal folding."""
    name = "mul_by_reciprocal"
    category = CAT_CONSTANT
    target_optimization = "reciprocal_constant_folding"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "mbr")
        c = np.array([5.05], dtype=np.float32)
        rc = 1.0 / c

        mul_o = f"{p}_mul"; out = f"{p}_out"
        bias = np.zeros(ctx.shape, dtype=np.float32)
        nodes = [
            helper.make_node("Mul", [input_name, f"{p}_rc"], [mul_o]),
            helper.make_node("Add", [mul_o, f"{p}_bias"], [out]),
        ]
        inits = [numpy_helper.from_array(rc,   f"{p}_rc"),
                 numpy_helper.from_array(bias, f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 15. Log(Exp(x)) cancel pair ───────────────────────────────────────────
class LogExpCancel(OTP):
    """Log(Exp(x)) = x (assuming no overflow). Tests exp/log cancellation."""
    name = "log_exp_cancel"
    category = CAT_CONSTANT
    target_optimization = "log_exp_elimination"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "lec")
        # Clip input to avoid overflow in Exp
        lo = np.array([-10.0], dtype=np.float32)
        hi = np.array([10.0],  dtype=np.float32)
        scale = np.ones(ctx.shape, dtype=np.float32)

        cl_o = f"{p}_cl"; ex_o = f"{p}_ex"; lg_o = f"{p}_lg"; out = f"{p}_out"
        nodes = [
            helper.make_node("Clip", [input_name, f"{p}_lo", f"{p}_hi"], [cl_o]),
            helper.make_node("Exp",  [cl_o], [ex_o]),
            helper.make_node("Log",  [ex_o], [lg_o]),
            helper.make_node("Mul",  [lg_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(lo,    f"{p}_lo"),
                 numpy_helper.from_array(hi,    f"{p}_hi"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 16. Learned temperature scalar ───────────────────────────────────────
class LearnedTemperatureScale(OTP):
    """x / temperature: logit scaling used in contrastive learning (CLIP-style)."""
    name = "learned_temperature_scale"
    category = CAT_CONSTANT
    target_optimization = "scalar_const_fold"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "lts")
        temp = np.array([0.505], dtype=np.float32)
        bias = np.zeros(ctx.shape, dtype=np.float32)

        div_o = f"{p}_div"; out = f"{p}_out"
        nodes = [
            helper.make_node("Div", [input_name, f"{p}_temp"], [div_o]),
            helper.make_node("Add", [div_o, f"{p}_bias"], [out]),
        ]
        inits = [numpy_helper.from_array(temp, f"{p}_temp"),
                 numpy_helper.from_array(bias, f"{p}_bias")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


ALL_CONSTANT_PATTERNS = [
    ConstantAddMul(),
    IdentityChain(),
    RedundantReshape(),
    SelfSubZero(),
    DivByConstant(),
    PowCanonical(),
    CastRoundTrip(),
    WhereConstCond(),
    # New patterns
    SqrtReciprocalMul(),
    TransposeInverseCancel(),
    ReshapeReshapeCancel(),
    SliceFullRange(),
    PadSliceNoop(),
    MulByReciprocal(),
    LogExpCancel(),
    LearnedTemperatureScale(),
]
