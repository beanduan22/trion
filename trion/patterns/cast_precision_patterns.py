"""
Cast-precision patterns.

Target the bf16↔fp32 and fp16↔fp32 round-trip-elide bug class documented in:

* pytorch#179561  (Inductor elides fp32→bf16→fp32)
* TF-XLA AlgebraicSimplifier  (verified in github_tfxla_001_bf16_cast_elide.py)
* google/jax  same XLA pass     (github_jax_001_bf16_cast_elide.py)

Plus:

* fp16 matmul chain — fp16 accumulation diverges from fp32 for large K.
* Commutative-reorder canary — (a+b)+c vs a+(b+c) with magnitudes chosen to
  catch compilers that blindly apply floating-point associativity.
* int32 truncation round-trip — Cast(fp32→int32→fp32) that compilers may
  incorrectly elide when they assume the fp32 input is already integer-valued.

Each pattern emits fp32 in, fp32 out, so it composes inside a fp32 graph.
The bf16/fp16 intermediate tensors are expressed directly via Cast with
``to=TensorProto.BFLOAT16`` (or FLOAT16). Downstream arithmetic around the
cast pair is deliberately present so the compiler sees the pair as an
interior simplification candidate.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper

from ..generation.context import StructuralContext
from .base import CAT_CONSTANT, CAT_FUSION, OTP, PatternInstance


# ---------------------------------------------------------------------------
# Cast roundtrip (bf16 / fp16 / int32)
# ---------------------------------------------------------------------------


class _CastRoundTrip(OTP):
    """
    x + eps  →  Cast(→LOW)  →  Cast(→fp32)  →  Add(*1.0)

    Wrapping arithmetic (both sides of the cast pair) ensures the compiler
    treats the two casts as an interior rewrite candidate rather than
    boundary IO casts.
    """

    category: str = CAT_CONSTANT
    target_optimization: str = "cast_roundtrip_elide"
    low_tensor_proto: int = TensorProto.BFLOAT16
    eps_value: float = 1.23e-7  # below bf16 resolution → bf16 rounds to 0

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "cp_rt")
        eps = np.array(self.eps_value, dtype=np.float32)
        one = np.array(1.0, dtype=np.float32)
        added = f"{p}_added"
        low = f"{p}_low"
        back = f"{p}_back"
        out = f"{p}_out"
        nodes = [
            helper.make_node("Add", [input_name, f"{p}_eps"], [added]),
            helper.make_node("Cast", [added], [low], to=self.low_tensor_proto),
            helper.make_node("Cast", [low], [back], to=TensorProto.FLOAT),
            helper.make_node("Mul", [back, f"{p}_one"], [out]),
        ]
        inits = [
            numpy_helper.from_array(eps, f"{p}_eps"),
            numpy_helper.from_array(one, f"{p}_one"),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_cast_rt(*, tag: str, low_proto: int, eps: float) -> type:
    name = f"cp_roundtrip_{tag}"
    return type(
        f"CastRT_{tag}",
        (_CastRoundTrip,),
        {
            "name": name,
            "low_tensor_proto": low_proto,
            "eps_value": eps,
            "target_optimization": f"cast_roundtrip_{tag}",
        },
    )


_CAST_RT_COMBOS: List[type] = [
    # bf16 has 7 mantissa bits → ~2^-7 ≈ 7.8e-3 relative resolution.
    _make_cast_rt(tag="bf16_tiny",  low_proto=TensorProto.BFLOAT16, eps=1.23e-7),
    _make_cast_rt(tag="bf16_small", low_proto=TensorProto.BFLOAT16, eps=1e-4),
    _make_cast_rt(tag="bf16_mid",   low_proto=TensorProto.BFLOAT16, eps=1e-2),
    # fp16 has 10 mantissa bits → resolution ~9.77e-4
    _make_cast_rt(tag="fp16_tiny",  low_proto=TensorProto.FLOAT16,  eps=1.23e-7),
    _make_cast_rt(tag="fp16_small", low_proto=TensorProto.FLOAT16,  eps=1e-4),
    # int32 truncation roundtrip — Cast(fp32→int32→fp32) drops fractional.
    # Use an eps whose magnitude can truncate to 0 if the compiler elides.
    _make_cast_rt(tag="int32_trunc", low_proto=TensorProto.INT32, eps=0.5),
    _make_cast_rt(tag="int32_trunc_neg", low_proto=TensorProto.INT32, eps=-0.5),
]


# ---------------------------------------------------------------------------
# Cast around an activation (Cast(fp32→fp16)→Relu→Cast(fp16→fp32))
# ---------------------------------------------------------------------------


class _CastActivation(OTP):
    category: str = CAT_FUSION
    target_optimization: str = "cast_activation_precision"
    low_tensor_proto: int = TensorProto.FLOAT16
    activation: str = "Relu"

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "cp_ca")
        low = f"{p}_low"
        act = f"{p}_act"
        back = f"{p}_back"
        nodes = [
            helper.make_node("Cast", [input_name], [low], to=self.low_tensor_proto),
            helper.make_node(self.activation, [low], [act]),
            helper.make_node("Cast", [act], [back], to=TensorProto.FLOAT),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            [],
            input_name,
            back,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_cast_act(*, tag: str, low_proto: int, activation: str) -> type:
    name = f"cp_cast_{tag}_{activation.lower()}"
    return type(
        f"CastAct_{tag}_{activation}",
        (_CastActivation,),
        {
            "name": name,
            "low_tensor_proto": low_proto,
            "activation": activation,
        },
    )


_CAST_ACT_COMBOS: List[type] = [
    _make_cast_act(tag="fp16", low_proto=TensorProto.FLOAT16,  activation="Relu"),
    _make_cast_act(tag="fp16", low_proto=TensorProto.FLOAT16,  activation="Sigmoid"),
    _make_cast_act(tag="fp16", low_proto=TensorProto.FLOAT16,  activation="Tanh"),
    # NOTE: bf16 activations are intentionally omitted — ORT CPU has no bf16
    # implementation of Relu/Sigmoid/Tanh, and Gelu is opset≥20. The bf16
    # cast-elide bug class is still covered by the _CAST_RT_COMBOS pairs
    # above (bf16_tiny / bf16_small / bf16_mid), whose arithmetic is fp32.
]


# ---------------------------------------------------------------------------
# fp16 MatMul chain — accumulation-order-sensitive
# ---------------------------------------------------------------------------


class _Fp16MatMulChain(OTP):
    """
    Cast(fp32→fp16) → MatMul(W_fp16) → Add(b_fp16) → Cast(fp16→fp32)

    The weight is (K, N) fp16. For large K, fp16 accumulation order diverges
    significantly from the fp32 reference — this is the ``cross_openvino_fp16
    _matmul_add`` scenario.
    """

    category: str = CAT_FUSION
    target_optimization: str = "fp16_matmul_accumulation"
    n: int = 512

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.rank == 2 and ctx.dtype == "float32" and ctx.shape[1] >= 8

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "cp_fm")
        N = self.n
        K = ctx.shape[1]
        # Deterministic weight at full fp32 precision, cast down to fp16 init.
        seed = int((K * 65537 + N * 131 + 11) % (2 ** 31))
        w_rng = np.random.default_rng(seed)
        w_f32 = (w_rng.standard_normal((K, N)) *
                 np.float32(np.sqrt(2.0 / (K + N)))).astype(np.float32)
        w_f16 = w_f32.astype(np.float16)
        b_f16 = np.zeros(N, dtype=np.float16)
        low = f"{p}_in_f16"
        mm = f"{p}_mm"
        add = f"{p}_add"
        out = f"{p}_out"
        nodes = [
            helper.make_node("Cast", [input_name], [low], to=TensorProto.FLOAT16),
            helper.make_node("MatMul", [low, f"{p}_w"], [mm]),
            helper.make_node("Add", [mm, f"{p}_b"], [add]),
            helper.make_node("Cast", [add], [out], to=TensorProto.FLOAT),
        ]
        inits = [
            numpy_helper.from_array(w_f16, f"{p}_w"),
            numpy_helper.from_array(b_f16, f"{p}_b"),
        ]
        batch = ctx.shape[0]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            out,
            StructuralContext(2, [batch, N], ctx.dtype, ctx.layout),
        )


def _make_fp16_mm(*, n: int) -> type:
    name = f"cp_fp16_matmul_n{n}"
    return type(
        f"Fp16Matmul_n{n}",
        (_Fp16MatMulChain,),
        {
            "name": name,
            "n": n,
            "target_optimization": "fp16_matmul_accumulation",
        },
    )


_FP16_MM_COMBOS: List[type] = [
    _make_fp16_mm(n=128),
    _make_fp16_mm(n=512),
    _make_fp16_mm(n=1024),
    _make_fp16_mm(n=2048),
]


# ---------------------------------------------------------------------------
# Commutativity canary: ((a+b)+c) - (a+(b+c))
# ---------------------------------------------------------------------------


class _CommutativeReorderCanary(OTP):
    """
    In fp16 (or bf16), the two parenthesisations of ``a+b+c`` disagree when
    ``a`` and ``b`` cancel but ``c`` is much smaller. A correct compiler must
    NOT reorder; ``expected=0``. A compiler that reorders yields non-zero.

    We express both orderings explicitly so the graph fixes the algebra, and
    downstream arithmetic links them back into the fp32 stream.
    """

    category: str = CAT_FUSION
    target_optimization: str = "assoc_reorder_fp16"
    low_tensor_proto: int = TensorProto.FLOAT16
    a_val: float = 1e4
    b_val: float = -1e4
    c_val: float = 1.0

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "cp_ass")
        a = np.array(self.a_val, dtype=np.float32)
        b = np.array(self.b_val, dtype=np.float32)
        c = np.array(self.c_val, dtype=np.float32)
        # Compute each ordering in LOW precision and subtract.
        a_low = f"{p}_a_low"
        b_low = f"{p}_b_low"
        c_low = f"{p}_c_low"
        # Ordering 1: (a+b)+c
        ab = f"{p}_ab"
        abc = f"{p}_abc"
        # Ordering 2: a+(b+c)
        bc = f"{p}_bc"
        abc2 = f"{p}_abc2"
        diff = f"{p}_diff"
        scaled = f"{p}_scaled"
        out = f"{p}_out"
        # To get the low-precision rounding while remaining executable on ORT
        # CPU (which lacks bf16 arithmetic), round each operand via
        # Cast(LOW) → Cast(fp32) and then perform the Add/Sub in fp32. The
        # compiler's cast-pair elision bug would eliminate the intermediate
        # round-trip, reinstating full fp32 precision and masking the
        # associativity signal.
        a_f = f"{p}_a_f"
        b_f = f"{p}_b_f"
        c_f = f"{p}_c_f"
        nodes = [
            helper.make_node("Cast", [f"{p}_a"], [a_low], to=self.low_tensor_proto),
            helper.make_node("Cast", [a_low], [a_f], to=TensorProto.FLOAT),
            helper.make_node("Cast", [f"{p}_b"], [b_low], to=self.low_tensor_proto),
            helper.make_node("Cast", [b_low], [b_f], to=TensorProto.FLOAT),
            helper.make_node("Cast", [f"{p}_c"], [c_low], to=self.low_tensor_proto),
            helper.make_node("Cast", [c_low], [c_f], to=TensorProto.FLOAT),
            helper.make_node("Add", [a_f, b_f], [ab]),
            helper.make_node("Add", [ab, c_f], [abc]),
            helper.make_node("Add", [b_f, c_f], [bc]),
            helper.make_node("Add", [a_f, bc], [abc2]),
            helper.make_node("Sub", [abc, abc2], [diff]),
            helper.make_node("Mul", [diff, f"{p}_eps"], [scaled]),
            helper.make_node("Add", [input_name, scaled], [out]),
        ]
        inits = [
            numpy_helper.from_array(a, f"{p}_a"),
            numpy_helper.from_array(b, f"{p}_b"),
            numpy_helper.from_array(c, f"{p}_c"),
            numpy_helper.from_array(np.array(1e-6, dtype=np.float32), f"{p}_eps"),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_assoc(*, tag: str, low_proto: int,
                a: float, b: float, c: float) -> type:
    name = f"cp_assoc_{tag}"
    return type(
        f"AssocReorder_{tag}",
        (_CommutativeReorderCanary,),
        {
            "name": name,
            "low_tensor_proto": low_proto,
            "a_val": a,
            "b_val": b,
            "c_val": c,
        },
    )


_ASSOC_COMBOS: List[type] = [
    _make_assoc(tag="fp16_1e4",  low_proto=TensorProto.FLOAT16,
                a=1e4, b=-1e4, c=1.0),
    _make_assoc(tag="bf16_1e8",  low_proto=TensorProto.BFLOAT16,
                a=1e8, b=-1e8, c=1.0),
    _make_assoc(tag="fp16_1e3",  low_proto=TensorProto.FLOAT16,
                a=1e3, b=-1e3, c=1e-3),
    _make_assoc(tag="bf16_1e4",  low_proto=TensorProto.BFLOAT16,
                a=1e4, b=-1e4, c=1.0),
]


# ---------------------------------------------------------------------------
# Public surfaces
# ---------------------------------------------------------------------------

_ALL_CLASSES: List[type] = (
    _CAST_RT_COMBOS + _CAST_ACT_COMBOS + _FP16_MM_COMBOS + _ASSOC_COMBOS
)

ALL_CAST_PRECISION_PATTERNS: List[OTP] = [cls() for cls in _ALL_CLASSES]

CAST_PRECISION_BY_CATEGORY: dict[str, List[OTP]] = {
    CAT_CONSTANT: [
        p for p in ALL_CAST_PRECISION_PATTERNS if p.category == CAT_CONSTANT
    ],
    CAT_FUSION: [
        p for p in ALL_CAST_PRECISION_PATTERNS if p.category == CAT_FUSION
    ],
}
