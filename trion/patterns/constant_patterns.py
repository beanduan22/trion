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
        c1 = rng.normal(0, 0.1, ctx.shape).astype(np.float32)
        c2 = np.abs(rng.normal(1, 0.1, ctx.shape)).astype(np.float32) + 0.01

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
        bias = rng.normal(0, 0.1, ctx.shape).astype(np.float32)
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
        c = np.array([float(rng.uniform(0.5, 10.0))], dtype=np.float32)
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
        other = rng.normal(0, 0.1, ctx.shape).astype(np.float32)

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


ALL_CONSTANT_PATTERNS = [
    ConstantAddMul(),
    IdentityChain(),
    RedundantReshape(),
    SelfSubZero(),
    DivByConstant(),
    PowCanonical(),
    CastRoundTrip(),
    WhereConstCond(),
]
