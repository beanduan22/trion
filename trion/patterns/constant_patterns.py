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
        # sub(x, x) = 0; then add(0, input) = input.
        # Compiler should fold Sub(x,x)→0 and then Add(0,x)→x (identity).
        sub_o = f"{p}_sub"; out = f"{p}_out"
        nodes = [
            helper.make_node("Sub", [input_name, input_name], [sub_o]),
            helper.make_node("Add", [sub_o, input_name], [out]),
        ]
        inits = []
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


# ── 17. Add(x, 0) identity ───────────────────────────────────────────────
class AddZeroIdentity(OTP):
    """Add(x, 0): should be eliminated as identity by constant folding."""
    name = "add_zero_identity"
    category = CAT_CONSTANT
    target_optimization = "add_zero_elimination"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "azi")
        zeros = np.zeros(ctx.shape, dtype=np.float32)
        scale = np.full(ctx.shape, 1.01, dtype=np.float32)

        add_o = f"{p}_add"; out = f"{p}_out"
        nodes = [
            helper.make_node("Add", [input_name, f"{p}_zeros"], [add_o]),
            helper.make_node("Mul", [add_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(zeros, f"{p}_zeros"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 18. Cast float32 → int32 → float32 (integer round-trip) ──────────────
class CastIntRoundTrip(OTP):
    """float32 → int32 → float32: tests integer cast precision handling."""
    name = "cast_int_roundtrip"
    category = CAT_CONSTANT
    target_optimization = "cast_int_elimination"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        import onnx
        p = self._p(node_id, "cirt")
        # Clip to safe integer range first to avoid overflow
        lo    = np.array([-100.0], dtype=np.float32)
        hi    = np.array([100.0],  dtype=np.float32)
        scale = np.full(ctx.shape, 0.01, dtype=np.float32)

        cl_o  = f"{p}_cl";  i32_o = f"{p}_i32"; out = f"{p}_out"
        nodes = [
            helper.make_node("Clip", [input_name, f"{p}_lo", f"{p}_hi"], [cl_o]),
            helper.make_node("Cast", [cl_o], [i32_o],
                             to=onnx.TensorProto.INT32),
            helper.make_node("Cast", [i32_o], [f"{p}_f32"],
                             to=onnx.TensorProto.FLOAT),
            helper.make_node("Mul",  [f"{p}_f32", f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(lo,    f"{p}_lo"),
                 numpy_helper.from_array(hi,    f"{p}_hi"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 19. Mul(x, 0) → zero-out ─────────────────────────────────────────────
class MulZeroElim(OTP):
    """Mul(x, 0) + Add(result, x): compiler should fold mul-by-zero then add."""
    name = "mul_zero_elim"
    category = CAT_CONSTANT
    target_optimization = "mul_zero_constant_fold"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "mze")
        zeros = np.zeros(ctx.shape, dtype=np.float32)
        scale = np.full(ctx.shape, 1.05, dtype=np.float32)

        mz_o = f"{p}_mz"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul", [input_name, f"{p}_zeros"], [mz_o]),
            helper.make_node("Add", [mz_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(zeros, f"{p}_zeros"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 20. Neg(Neg(x)) cancel pair ──────────────────────────────────────────
class NegNegCancel(OTP):
    """Neg(Neg(x)) = x: double negation elimination."""
    name = "neg_neg_cancel"
    category = CAT_CONSTANT
    target_optimization = "double_neg_elimination"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "nnc")
        scale = np.full(ctx.shape, 0.5, dtype=np.float32)

        n1_o = f"{p}_n1"; n2_o = f"{p}_n2"; out = f"{p}_out"
        nodes = [
            helper.make_node("Neg", [input_name], [n1_o]),
            helper.make_node("Neg", [n1_o], [n2_o]),
            helper.make_node("Mul", [n2_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 21. Expand → Squeeze (broadcast + reduce cancel) ─────────────────────
class ExpandSqueeze(OTP):
    """Expand + ReduceSum over broadcast dims: targets broadcast constant folding."""
    name = "expand_squeeze"
    category = CAT_CONSTANT
    target_optimization = "expand_reduce_fold"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32" and ctx.shape[0] >= 1

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "esq")
        B, D = ctx.shape
        factor = 3
        expand_shape = np.array([B, factor, D], dtype=np.int64)
        scale = np.full([B, D], 1.0 / factor, dtype=np.float32)

        # opset 13+: ReduceSum takes `axes` as a 1-D int64 input, not an attribute.
        red_axes = np.array([1], dtype=np.int64)
        exp_o = f"{p}_exp"; red_o = f"{p}_red"; out = f"{p}_out"
        nodes = [
            helper.make_node("Expand", [input_name, f"{p}_esh"], [exp_o]),
            helper.make_node("ReduceSum", [exp_o, f"{p}_raxes"], [red_o], keepdims=0),
            helper.make_node("Mul", [red_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(expand_shape, f"{p}_esh"),
                 numpy_helper.from_array(red_axes,     f"{p}_raxes"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [B, D], ctx.dtype, ctx.layout))


# ── 22. MaxPool → GlobalAvgPool (spatial collapse with const fold) ────────
class MaxPoolGlobalAvg(OTP):
    """MaxPool(kernel=HxW) + GlobalAveragePool: spatial reduction constant folding."""
    name = "maxpool_global_avg"
    category = CAT_CONSTANT
    target_optimization = "spatial_reduction_fold"

    def is_compatible(self, ctx):
        return ctx.rank == 4 and ctx.dtype == "float32" and ctx.shape[2] >= 4 and ctx.shape[3] >= 4

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "mpga")
        kH, kW = min(H, 4), min(W, 4)
        oH, oW = H // kH, W // kW

        mp_o = f"{p}_mp"; gap_o = f"{p}_gap"; out = f"{p}_out"
        scale = np.ones([N, C, 1, 1], dtype=np.float32) * 1.01
        nodes = [
            helper.make_node("MaxPool", [input_name], [mp_o],
                             kernel_shape=[kH, kW], strides=[kH, kW]),
            helper.make_node("GlobalAveragePool", [mp_o], [gap_o]),
            helper.make_node("Mul", [gap_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(4, [N, C, 1, 1], ctx.dtype, ctx.layout))


# ── 23. Mul(x, x) → Pow(x, 2) equivalence ────────────────────────────────
class MulSelfToPow(OTP):
    """Mul(x,x) + Sqrt: compiler may rewrite Mul(x,x) to Pow(x,2) then simplify."""
    name = "mul_self_to_pow"
    category = CAT_CONSTANT
    target_optimization = "mul_self_pow_rewrite"

    def is_compatible(self, ctx):
        return ctx.dtype == "float32"

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "msp")
        eps = np.array([1e-6], dtype=np.float32)
        scale = np.ones(ctx.shape, dtype=np.float32)

        sq_o = f"{p}_sq"; add_o = f"{p}_add"; sq2_o = f"{p}_sq2"; out = f"{p}_out"
        nodes = [
            helper.make_node("Mul",  [input_name, input_name], [sq_o]),
            helper.make_node("Add",  [sq_o, f"{p}_eps"], [add_o]),
            helper.make_node("Sqrt", [add_o], [sq2_o]),
            helper.make_node("Mul",  [sq2_o, f"{p}_scale"], [out]),
        ]
        inits = [numpy_helper.from_array(eps,   f"{p}_eps"),
                 numpy_helper.from_array(scale, f"{p}_scale")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(ctx.rank, list(ctx.shape),
                                                 ctx.dtype, ctx.layout))


# ── 24. CumSum along last axis ────────────────────────────────────────────
class CumSumNormalize(OTP):
    """CumSum → Div by length: prefix-sum normalization, tests scan lowering."""
    name = "cumsum_normalize"
    category = CAT_CONSTANT
    target_optimization = "cumsum_folding"

    def is_compatible(self, ctx):
        return ctx.rank == 2 and ctx.dtype == "float32" and ctx.shape[1] >= 4

    def instantiate(self, input_name, ctx, rng, node_id):
        p = self._p(node_id, "csn")
        B, L = ctx.shape
        axis_t = np.array(1, dtype=np.int64)
        lengths = np.arange(1, L + 1, dtype=np.float32).reshape(1, L)

        cs_o = f"{p}_cs"; out = f"{p}_out"
        nodes = [
            helper.make_node("CumSum", [input_name, f"{p}_axis"], [cs_o],
                             exclusive=0, reverse=0),
            helper.make_node("Div", [cs_o, f"{p}_len"], [out]),
        ]
        inits = [numpy_helper.from_array(axis_t, f"{p}_axis"),
                 numpy_helper.from_array(lengths, f"{p}_len")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [B, L], ctx.dtype, ctx.layout))


# ── 25. Flatten → Gemm (FC layer via constant weight) ────────────────────
class FlattenGemm(OTP):
    """Flatten + Gemm: standard FC pattern, triggers weight const-folding."""
    name = "flatten_gemm"
    category = CAT_CONSTANT
    target_optimization = "gemm_weight_folding"

    def is_compatible(self, ctx):
        # Cap C*H*W so the dense weight stays under a few MB.
        if ctx.rank != 4 or ctx.dtype != "float32":
            return False
        _, C, H, W = ctx.shape
        return C * H * W <= 4096

    def instantiate(self, input_name, ctx, rng, node_id):
        N, C, H, W = ctx.shape
        p = self._p(node_id, "fgm")
        in_f = C * H * W
        out_f = max(16, in_f // 4)
        seed = int((in_f * 65537 + out_f) % (2 ** 31))
        rng_local = np.random.default_rng(seed)
        scale_v = np.float32(np.sqrt(2.0 / (in_f + out_f)))
        W_arr = (rng_local.standard_normal((out_f, in_f)) * scale_v).astype(np.float32)
        b_arr = np.zeros(out_f, dtype=np.float32)

        fl_o = f"{p}_fl"; out = f"{p}_out"
        nodes = [
            helper.make_node("Flatten", [input_name], [fl_o], axis=1),
            helper.make_node("Gemm", [fl_o, f"{p}_W", f"{p}_b"], [out],
                             transB=1),
        ]
        inits = [numpy_helper.from_array(W_arr, f"{p}_W"),
                 numpy_helper.from_array(b_arr, f"{p}_b")]
        return PatternInstance(self.name, self.category, nodes, inits,
                               input_name, out,
                               StructuralContext(2, [N, out_f], ctx.dtype, ctx.layout))


ALL_CONSTANT_PATTERNS = [
    ConstantAddMul(),
    IdentityChain(),
    RedundantReshape(),
    SelfSubZero(),
    DivByConstant(),
    PowCanonical(),
    CastRoundTrip(),
    WhereConstCond(),
    SqrtReciprocalMul(),
    TransposeInverseCancel(),
    ReshapeReshapeCancel(),
    SliceFullRange(),
    PadSliceNoop(),
    MulByReciprocal(),
    LogExpCancel(),
    LearnedTemperatureScale(),
    AddZeroIdentity(),
    CastIntRoundTrip(),
    MulZeroElim(),
    NegNegCancel(),
    ExpandSqueeze(),
    MaxPoolGlobalAvg(),
    MulSelfToPow(),
    CumSumNormalize(),
    FlattenGemm(),
]
