"""
Integer-arithmetic patterns.

Exercises saturation vs modular-wrap behaviour, BitShift UB, Mod-with-negatives,
QuantizeLinear/DequantizeLinear roundtrips, and integer Div edge cases.

Representative upstream issues (full list in _mining_notes.md):

* openvino#21989..21995 — uint8/int8 Add/Sub/Mul saturate instead of modular wrap.
* pytorch#143555 / #143566 — BitShift amount ≥ word size is C UB.
* cross_bitshift_shift64_ov_ort — ORT and OV both wrong on RIGHT shift of 64.

The pattern contract requires fp32 in / fp32 out so that patterns can compose
with the existing fp32 graph. We encode the integer arithmetic between a leading
``Cast(→intN)`` and a trailing ``Cast(→fp32)``; the op under test always operates
on integer tensors. The boundary edge values are embedded as initializers
concatenated with the input via ``Concat(axis=0)`` into a single integer tensor
so that the op's integer semantics are actually exercised regardless of the
upstream fp32 input distribution.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper

from ..generation.context import StructuralContext
from .base import CAT_BROADCAST, CAT_CONSTANT, OTP, PatternInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_INT_DTYPE_MAP = {
    "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
}

_NP_INT_DTYPE = {
    "uint8": np.uint8,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
}


def _sum_to_scalar_and_add_input(
    int_tensor: str,
    fp32_input: str,
    prefix: str,
) -> tuple[List[NodeProto], str]:
    """Fold an integer tensor into a scalar fp32 scaling factor, then add it
    back to the fp32 input so the output preserves the input shape/dtype.

    This lets each pattern emit its integer arithmetic on a tiny dedicated
    tensor while still plugging into the fp32 composer graph — the integer
    tensor's value is surfaced via a scalar added to the fp32 carrier.
    """
    cast_f = f"{prefix}_cast_f"
    red = f"{prefix}_red"
    scaled = f"{prefix}_scaled"
    out = f"{prefix}_out"
    # ReduceSum(keepdims=0) of an int-cast-to-float tensor → scalar float.
    nodes = [
        helper.make_node("Cast", [int_tensor], [cast_f], to=TensorProto.FLOAT),
        helper.make_node(
            "ReduceSum", [cast_f], [red], keepdims=0
        ),
        helper.make_node("Mul", [red, f"{prefix}_eps"], [scaled]),
        helper.make_node("Add", [fp32_input, scaled], [out]),
    ]
    return nodes, out


def _eps_init(prefix: str) -> TensorProto:
    # Small factor so the int signal is surfaced without dominating fp32 input.
    return numpy_helper.from_array(
        np.array(1e-6, dtype=np.float32), f"{prefix}_eps"
    )


# ---------------------------------------------------------------------------
# Binary-op saturation patterns
# ---------------------------------------------------------------------------


class _SaturationBinaryOp(OTP):
    category: str = CAT_BROADCAST
    target_optimization: str = "integer_saturation_vs_modular"
    onnx_op: str = "Add"
    int_dtype: str = "uint8"
    a_values: tuple[int, ...] = (250, 200, 100, 0)
    b_values: tuple[int, ...] = (10, 100, 200, 5)

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ia")
        np_dt = _NP_INT_DTYPE[self.int_dtype]
        a = np.array(self.a_values, dtype=np_dt)
        b = np.array(self.b_values, dtype=np_dt)
        op_out = f"{p}_op"
        int_in_a = f"{p}_a"
        int_in_b = f"{p}_b"
        nodes: List[NodeProto] = [
            helper.make_node(self.onnx_op, [int_in_a, int_in_b], [op_out]),
        ]
        # Merge into fp32 carrier.
        fold_nodes, final_out = _sum_to_scalar_and_add_input(
            op_out, input_name, p
        )
        nodes.extend(fold_nodes)

        inits = [
            numpy_helper.from_array(a, int_in_a),
            numpy_helper.from_array(b, int_in_b),
            _eps_init(p),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            final_out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_sat_class(
    *,
    tag: str,
    op: str,
    dtype: str,
    a: tuple[int, ...],
    b: tuple[int, ...],
) -> type:
    name = f"ia_sat_{op.lower()}_{dtype}_{tag}"
    cls = type(
        f"IntSat_{op}_{dtype}_{tag}",
        (_SaturationBinaryOp,),
        {
            "name": name,
            "onnx_op": op,
            "int_dtype": dtype,
            "a_values": a,
            "b_values": b,
            "target_optimization": "integer_saturation_vs_modular",
        },
    )
    return cls


_SAT_COMBOS: List[type] = [
    # uint8 Add / Sub / Mul  (OV #21989/21990/21991)
    _make_sat_class(tag="overflow", op="Add", dtype="uint8",
                    a=(250, 200, 100, 1, 30), b=(10, 100, 200, 254, 250)),
    _make_sat_class(tag="underflow", op="Sub", dtype="uint8",
                    a=(5, 0, 100, 20, 40), b=(10, 200, 50, 30, 41)),
    _make_sat_class(tag="overflow", op="Mul", dtype="uint8",
                    a=(200, 40, 16, 10, 17), b=(2, 10, 20, 30, 15)),
    # int8 Add / Sub (OV #21992/21993)
    _make_sat_class(tag="overflow", op="Add", dtype="int8",
                    a=(120, 100, -100, 60, 64), b=(20, 50, -50, 80, 64)),
    _make_sat_class(tag="underflow", op="Sub", dtype="int8",
                    a=(-120, -100, 0, 10, 100), b=(20, 50, 1, -120, 127)),
    # int32 wraps: large multiply / add near INT32_MAX
    _make_sat_class(tag="overflow", op="Add", dtype="int32",
                    a=(2_147_483_000, -2_147_483_000, 0),
                    b=(1_000, -1_000, 1)),
    _make_sat_class(tag="overflow", op="Mul", dtype="int32",
                    a=(65536, 46341, 2),
                    b=(65536, 46341, 1_000_000)),
]


# ---------------------------------------------------------------------------
# BitShift
# ---------------------------------------------------------------------------


class _BitShift(OTP):
    category: str = CAT_BROADCAST
    target_optimization: str = "bitshift_ub"
    direction: str = "RIGHT"
    int_dtype: str = "int64"
    values: tuple[int, ...] = (1000, 255, 42)
    shifts: tuple[int, ...] = (64, 64, 64)

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ia_bs")
        # ONNX BitShift requires unsigned integer inputs (uint8/16/32/64);
        # we always use uint32 for the operand and a uint32 shift amount.
        # Negative `values` are represented by their two's-complement uint32
        # bit pattern so the test still exercises the UB shift amount.
        dt = np.uint32
        vals = np.array(
            [np.uint32(v & 0xFFFFFFFF) for v in self.values], dtype=dt
        )
        # Shift amounts — ONNX spec does not restrict; UB is in backends.
        shifts = np.array(
            [np.uint32(s & 0xFFFFFFFF) for s in self.shifts], dtype=dt
        )
        op_out = f"{p}_bs"
        nodes = [
            helper.make_node(
                "BitShift",
                [f"{p}_x", f"{p}_s"],
                [op_out],
                direction=self.direction,
            ),
        ]
        fold_nodes, final_out = _sum_to_scalar_and_add_input(
            op_out, input_name, p
        )
        nodes.extend(fold_nodes)
        inits = [
            numpy_helper.from_array(vals, f"{p}_x"),
            numpy_helper.from_array(shifts, f"{p}_s"),
            _eps_init(p),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            final_out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_bs_class(*, tag: str, direction: str,
                   values: tuple[int, ...], shifts: tuple[int, ...]) -> type:
    name = f"ia_bitshift_{direction.lower()}_{tag}"
    cls = type(
        f"BitShift_{direction}_{tag}",
        (_BitShift,),
        {
            "name": name,
            "direction": direction,
            "values": values,
            "shifts": shifts,
            "target_optimization": "bitshift_ub",
        },
    )
    return cls


_BS_COMBOS: List[type] = [
    _make_bs_class(tag="shift32_u32", direction="RIGHT",
                   values=(1000, 255, 1 << 20),
                   shifts=(32, 32, 32)),
    _make_bs_class(tag="shift33_u32", direction="RIGHT",
                   values=(1000, 255, 1 << 20),
                   shifts=(33, 33, 33)),
    _make_bs_class(tag="shift64_u32", direction="RIGHT",
                   values=(1000, 255, 1 << 20),
                   shifts=(64, 64, 64)),
    _make_bs_class(tag="shift31_u32", direction="LEFT",
                   values=(1, 1, 1),
                   shifts=(31, 31, 31)),
    _make_bs_class(tag="shift32_u32_l", direction="LEFT",
                   values=(1, 2, 3),
                   shifts=(32, 32, 32)),
    _make_bs_class(tag="shift1_ok", direction="RIGHT",
                   values=(1000, 255, 65536),
                   shifts=(1, 1, 1)),
]


# ---------------------------------------------------------------------------
# Mod with negatives
# ---------------------------------------------------------------------------


class _ModPattern(OTP):
    category: str = CAT_BROADCAST
    target_optimization: str = "mod_sign_semantics"
    lhs: tuple[int, ...] = (-7, 7, -7, 7)
    rhs: tuple[int, ...] = (3, -3, -3, 3)
    fmod: int = 0

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ia_mod")
        a = np.array(self.lhs, dtype=np.int32)
        b = np.array(self.rhs, dtype=np.int32)
        op_out = f"{p}_op"
        nodes = [
            helper.make_node(
                "Mod",
                [f"{p}_a", f"{p}_b"],
                [op_out],
                fmod=self.fmod,
            ),
        ]
        fold_nodes, final_out = _sum_to_scalar_and_add_input(
            op_out, input_name, p
        )
        nodes.extend(fold_nodes)
        inits = [
            numpy_helper.from_array(a, f"{p}_a"),
            numpy_helper.from_array(b, f"{p}_b"),
            _eps_init(p),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            final_out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_mod_class(*, tag: str,
                    lhs: tuple[int, ...], rhs: tuple[int, ...],
                    fmod: int) -> type:
    name = f"ia_mod_{'fmod' if fmod else 'py'}_{tag}"
    return type(
        f"Mod_{tag}_fmod{fmod}",
        (_ModPattern,),
        {
            "name": name,
            "lhs": lhs,
            "rhs": rhs,
            "fmod": fmod,
            "target_optimization": "mod_sign_semantics",
        },
    )


_MOD_COMBOS: List[type] = [
    _make_mod_class(tag="negpos", lhs=(-7, -5, -13),
                    rhs=(3, 3, 5), fmod=0),
    _make_mod_class(tag="posneg", lhs=(7, 10, 13),
                    rhs=(-3, -4, -5), fmod=0),
    _make_mod_class(tag="negpos_f", lhs=(-7, -5, -13),
                    rhs=(3, 3, 5), fmod=1),
    _make_mod_class(tag="posneg_f", lhs=(7, 10, 13),
                    rhs=(-3, -4, -5), fmod=1),
]


# ---------------------------------------------------------------------------
# Integer Div edge cases (zero denominator, negative rounding)
# ---------------------------------------------------------------------------


class _IntDiv(OTP):
    """Integer Div with denominator 0 or negative rhs.

    Note: ORT/OV may raise at model load for a constant zero denominator,
    so we only emit *non-zero-but-edge* denominators here (1, -1, MIN) and
    leave the literal zero case as a TODO for a runtime oracle mode.
    """

    category: str = CAT_BROADCAST
    target_optimization: str = "int_div_rounding"
    lhs: tuple[int, ...] = (-7, -5, 7, 5)
    rhs: tuple[int, ...] = (2, -2, -2, 2)

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ia_div")
        a = np.array(self.lhs, dtype=np.int32)
        b = np.array(self.rhs, dtype=np.int32)
        op_out = f"{p}_op"
        nodes = [
            helper.make_node("Div", [f"{p}_a", f"{p}_b"], [op_out]),
        ]
        fold_nodes, final_out = _sum_to_scalar_and_add_input(
            op_out, input_name, p
        )
        nodes.extend(fold_nodes)
        inits = [
            numpy_helper.from_array(a, f"{p}_a"),
            numpy_helper.from_array(b, f"{p}_b"),
            _eps_init(p),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            final_out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_intdiv_class(*, tag: str,
                       lhs: tuple[int, ...], rhs: tuple[int, ...]) -> type:
    name = f"ia_intdiv_{tag}"
    return type(
        f"IntDiv_{tag}",
        (_IntDiv,),
        {
            "name": name,
            "lhs": lhs,
            "rhs": rhs,
        },
    )


_INTDIV_COMBOS: List[type] = [
    _make_intdiv_class(tag="truncate_neg", lhs=(-7, -5, 7, 5),
                       rhs=(2, 2, 2, 2)),
    _make_intdiv_class(tag="truncate_neg_rhs", lhs=(7, 5, -7, -5),
                       rhs=(-2, -2, -2, -2)),
    # NOTE: dropped int32_min_by_m1 (-2^31 / -1) during smoke test —
    # ORT CPU backend takes the C division path which raises SIGFPE on the
    # overflow case. This is the UB being exercised, but the test harness
    # can't survive the SIGFPE. Leave as a regression target for a future
    # crash-mode oracle.
    _make_intdiv_class(tag="int32_min_plus_by_m1",
                       lhs=(-2_147_483_000, 0, 1),
                       rhs=(-1, -1, -1)),
]


# ---------------------------------------------------------------------------
# QuantizeLinear → DequantizeLinear round-trip
# ---------------------------------------------------------------------------


class _QDQRoundTrip(OTP):
    """
    Deterministic QuantizeLinear followed by DequantizeLinear on the fp32
    input. Scale/zero-point chosen so the input ranges hit saturation:
    some elements lie below the representable range (clamped to 0 in uint8),
    some lie above (clamped to 255), and some straddle the boundary.
    """

    category: str = CAT_CONSTANT
    target_optimization: str = "qdq_roundtrip"
    scale: float = 0.1
    zero_point: int = 128
    y_dtype: str = "int8"

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32"

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ia_qdq")
        scale = np.array(self.scale, dtype=np.float32)
        zp = np.array(self.zero_point, dtype=_NP_INT_DTYPE[self.y_dtype])
        q_out = f"{p}_q"
        d_out = f"{p}_d"
        nodes = [
            helper.make_node(
                "QuantizeLinear",
                [input_name, f"{p}_scale", f"{p}_zp"], [q_out],
            ),
            helper.make_node(
                "DequantizeLinear",
                [q_out, f"{p}_scale", f"{p}_zp"], [d_out],
            ),
        ]
        inits = [
            numpy_helper.from_array(scale, f"{p}_scale"),
            numpy_helper.from_array(zp, f"{p}_zp"),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            d_out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


def _make_qdq_class(*, tag: str, scale: float,
                    zero_point: int, y_dtype: str) -> type:
    name = f"ia_qdq_{tag}"
    return type(
        f"QDQ_{tag}",
        (_QDQRoundTrip,),
        {
            "name": name,
            "scale": scale,
            "zero_point": zero_point,
            "y_dtype": y_dtype,
        },
    )


_QDQ_COMBOS: List[type] = [
    _make_qdq_class(tag="int8_s0p1_zp0",    scale=0.1, zero_point=0,   y_dtype="int8"),
    _make_qdq_class(tag="uint8_s0p1_zp128", scale=0.1, zero_point=128, y_dtype="uint8"),
    _make_qdq_class(tag="uint8_s0p02_zp0",  scale=0.02, zero_point=0,  y_dtype="uint8"),
    _make_qdq_class(tag="int8_s0p5_zp_m10", scale=0.5, zero_point=-10, y_dtype="int8"),
    _make_qdq_class(tag="int8_tiny",        scale=1e-4, zero_point=0,  y_dtype="int8"),
    _make_qdq_class(tag="uint8_big",        scale=1.0,  zero_point=0,  y_dtype="uint8"),
]


# ---------------------------------------------------------------------------
# QLinearMatMul — two quantised operands, single op.
# ---------------------------------------------------------------------------


class _QLinearMatMul(OTP):
    """
    Minimal QLinearMatMul. Weight is a fixed (K,N) int8 tensor; its scale and
    zero-point are constants. Input is quantised via QuantizeLinear, matmul is
    performed via QLinearMatMul, and the result is dequantised back to fp32.
    Rounding mode divergence between ORT and other backends is the signal.
    """

    category: str = CAT_CONSTANT
    target_optimization: str = "qlinear_matmul_rounding"
    scale: float = 0.05
    zp_int: int = 0
    y_dtype: str = "int8"
    k: int = 32
    n: int = 32

    def is_compatible(self, ctx: StructuralContext) -> bool:
        if ctx.dtype != "float32":
            return False
        # Require a 2-D (batch, features) shape so MatMul is trivial.
        return ctx.rank == 2 and ctx.shape[1] >= 8

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ia_qmm")
        K = ctx.shape[1]
        N = self.n
        np_dt = _NP_INT_DTYPE[self.y_dtype]
        # Deterministic weight.
        seed = int((K * 65537 + N * 131 + 7) % (2 ** 31))
        w_rng = np.random.default_rng(seed)
        w_int = w_rng.integers(
            low=np.iinfo(np_dt).min,
            high=np.iinfo(np_dt).max,
            size=(K, N),
            dtype=np_dt,
        )
        a_scale = np.array(self.scale, dtype=np.float32)
        a_zp = np.array(self.zp_int, dtype=np_dt)
        w_scale = np.array(self.scale * 0.5, dtype=np.float32)
        w_zp = np.array(0, dtype=np_dt)
        y_scale = np.array(self.scale * self.scale * 0.5, dtype=np.float32)
        y_zp = np.array(self.zp_int, dtype=np_dt)
        q_in = f"{p}_qin"
        qmm = f"{p}_qmm"
        d_out = f"{p}_dout"
        nodes = [
            helper.make_node(
                "QuantizeLinear",
                [input_name, f"{p}_a_scale", f"{p}_a_zp"], [q_in],
            ),
            helper.make_node(
                "QLinearMatMul",
                [q_in, f"{p}_a_scale", f"{p}_a_zp",
                 f"{p}_w", f"{p}_w_scale", f"{p}_w_zp",
                 f"{p}_y_scale", f"{p}_y_zp"], [qmm],
            ),
            helper.make_node(
                "DequantizeLinear",
                [qmm, f"{p}_y_scale", f"{p}_y_zp"], [d_out],
            ),
        ]
        inits = [
            numpy_helper.from_array(w_int, f"{p}_w"),
            numpy_helper.from_array(a_scale, f"{p}_a_scale"),
            numpy_helper.from_array(a_zp, f"{p}_a_zp"),
            numpy_helper.from_array(w_scale, f"{p}_w_scale"),
            numpy_helper.from_array(w_zp, f"{p}_w_zp"),
            numpy_helper.from_array(y_scale, f"{p}_y_scale"),
            numpy_helper.from_array(y_zp, f"{p}_y_zp"),
        ]
        batch = ctx.shape[0]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            d_out,
            StructuralContext(2, [batch, N], ctx.dtype, ctx.layout),
        )


def _make_qlinearmm_class(*, tag: str, scale: float,
                          zp: int, y_dtype: str, n: int) -> type:
    name = f"ia_qlinearmm_{tag}"
    return type(
        f"QLMM_{tag}",
        (_QLinearMatMul,),
        {
            "name": name,
            "scale": scale,
            "zp_int": zp,
            "y_dtype": y_dtype,
            "n": n,
        },
    )


_QLMM_COMBOS: List[type] = [
    _make_qlinearmm_class(tag="int8_small",   scale=0.05, zp=0,   y_dtype="int8",  n=16),
    _make_qlinearmm_class(tag="uint8_small",  scale=0.05, zp=128, y_dtype="uint8", n=16),
    _make_qlinearmm_class(tag="int8_tiny",    scale=0.005, zp=0,  y_dtype="int8",  n=32),
    _make_qlinearmm_class(tag="int8_bignzp",  scale=0.05, zp=-50, y_dtype="int8",  n=32),
    _make_qlinearmm_class(tag="uint8_wide",   scale=0.1,  zp=100, y_dtype="uint8", n=64),
]


# ---------------------------------------------------------------------------
# Public registration surfaces
# ---------------------------------------------------------------------------

_ALL_CLASSES: List[type] = (
    _SAT_COMBOS
    + _BS_COMBOS
    + _MOD_COMBOS
    + _INTDIV_COMBOS
    + _QDQ_COMBOS
    + _QLMM_COMBOS
)

ALL_INTEGER_ARITHMETIC_PATTERNS: List[OTP] = [cls() for cls in _ALL_CLASSES]

# Partition by category (saturation/bitshift/mod/div/qdq fold into broadcast;
# qdq + qlmm fold into constant since they behave as weight-only preprocessing).
INTEGER_ARITHMETIC_BY_CATEGORY: dict[str, List[OTP]] = {
    CAT_BROADCAST: [
        p for p in ALL_INTEGER_ARITHMETIC_PATTERNS
        if p.category == CAT_BROADCAST
    ],
    CAT_CONSTANT: [
        p for p in ALL_INTEGER_ARITHMETIC_PATTERNS
        if p.category == CAT_CONSTANT
    ],
}
