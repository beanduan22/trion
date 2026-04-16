"""
IEEE-754 edge-input patterns.

Each pattern injects one of {NaN, +Inf, -Inf, 0, -0, subnormal} into the tensor
flowing through an op that is spec-sensitive to that value. The injection site
is always a constant tensor combined with the fp32 input via Concat along the
last axis — the downstream composer still sees a rank-preserving, fp32-shaped
output.

Representative upstream issues (see _mining_notes.md):

* openvino#21994 — Relu(NaN) → 0 instead of NaN propagation.
* openvino#21995 — Exp(NaN) → +Inf instead of NaN.
* openvino#21988 — ReduceLogSumExp overflows (missing max-subtract).
* onnxruntime#10588 — Sqrt(-0) dropping sign bit.
* onnxruntime#16502 — Reciprocal(0) sign of infinity.
* pytorch#94624 — subnormal flush-to-zero in Inductor CPU MKL path.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper

from ..generation.context import StructuralContext
from .base import CAT_BROADCAST, OTP, PatternInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concat_edge_with_input(
    input_name: str,
    prefix: str,
    edge_values: np.ndarray,
    target_shape: List[int],
) -> Tuple[List[NodeProto], List[TensorProto], str]:
    """
    Build a tensor whose trailing axis is extended by ``edge_values``:
        tmp = Concat([input, const_edge], axis=-1)

    The const is reshape-broadcast to match the prefix dims of the input so
    that Concat is legal. Returns (nodes, inits, concat_output_name). The
    caller is responsible for slicing back to the original shape at the end.
    """
    k = int(edge_values.shape[-1])
    # Build const of shape [1, 1, ..., k] broadcast to input-prefix dims.
    # We use a ConstantOfShape + Scatter-free strategy: pre-broadcast the const
    # to the prefix dims statically, using np.broadcast_to + copy.
    const_shape = list(target_shape[:-1]) + [k]
    prefix_shape = target_shape[:-1]
    # Broadcast the edge-values (shape [k]) to [*prefix_shape, k].
    broadcast_const = np.broadcast_to(
        edge_values.reshape([1] * len(prefix_shape) + [k]),
        const_shape,
    ).astype(np.float32).copy()

    const_name = f"{prefix}_edge_const"
    concat_out = f"{prefix}_concat"
    nodes = [
        helper.make_node(
            "Concat", [input_name, const_name], [concat_out], axis=-1
        ),
    ]
    inits = [numpy_helper.from_array(broadcast_const, const_name)]
    return nodes, inits, concat_out


def _slice_first_original(
    concat_out: str,
    prefix: str,
    original_last_dim: int,
) -> Tuple[List[NodeProto], List[TensorProto], str]:
    """Slice the leading ``original_last_dim`` elements along axis=-1."""
    starts = np.array([0], dtype=np.int64)
    ends = np.array([original_last_dim], dtype=np.int64)
    axes = np.array([-1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    out = f"{prefix}_slice"
    nodes = [
        helper.make_node(
            "Slice",
            [concat_out, f"{prefix}_starts", f"{prefix}_ends",
             f"{prefix}_axes", f"{prefix}_steps"],
            [out],
        )
    ]
    inits = [
        numpy_helper.from_array(starts, f"{prefix}_starts"),
        numpy_helper.from_array(ends, f"{prefix}_ends"),
        numpy_helper.from_array(axes, f"{prefix}_axes"),
        numpy_helper.from_array(steps, f"{prefix}_steps"),
    ]
    return nodes, inits, out


# ---------------------------------------------------------------------------
# Unary-op edge pattern
# ---------------------------------------------------------------------------


class _UnaryEdgePattern(OTP):
    """
    Generic unary-op edge-value pattern:

        concat = Concat([x, edge_const], axis=-1)
        y      = Op(concat)
        out    = Slice(y, 0, x.shape[-1], axis=-1)

    The slice preserves the original shape so the pattern chains like a
    standard unary op in the fp32 stream. The edge-value-bearing region of
    the tensor still flows through ``Op`` so compilers that mishandle
    IEEE-754 corner cases internally are exposed.
    """

    category: str = CAT_BROADCAST
    target_optimization: str = "edge_input"
    onnx_op: str = "Relu"
    edge_values: Tuple[float, ...] = (float("nan"),)
    # Extra attributes to pass to helper.make_node (e.g. Softmax axis).
    op_attrs: dict = {}

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32" and ctx.rank >= 1 and ctx.shape[-1] >= 1

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ei")
        orig_last = ctx.shape[-1]
        edge_arr = np.array(self.edge_values, dtype=np.float32)
        concat_nodes, concat_inits, concat_out = _concat_edge_with_input(
            input_name, p, edge_arr, list(ctx.shape)
        )
        op_out = f"{p}_op"
        op_node = helper.make_node(
            self.onnx_op, [concat_out], [op_out], **(self.op_attrs or {})
        )
        slice_nodes, slice_inits, final_out = _slice_first_original(
            op_out, p, orig_last
        )
        return PatternInstance(
            self.name,
            self.category,
            concat_nodes + [op_node] + slice_nodes,
            concat_inits + slice_inits,
            input_name,
            final_out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


# ---------------------------------------------------------------------------
# Binary-op edge pattern (Div by edge, Pow(0,0), Min/Max(NaN, x))
# ---------------------------------------------------------------------------


class _BinaryEdgePattern(OTP):
    """
    Similar structure to the unary case, but the op takes the concatenated
    tensor on one side and a broadcast-compatible edge constant on the other.
    """

    category: str = CAT_BROADCAST
    target_optimization: str = "edge_input_binary"
    onnx_op: str = "Div"
    edge_values: Tuple[float, ...] = (float("nan"),)
    rhs_value: float = 0.0

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32" and ctx.rank >= 1 and ctx.shape[-1] >= 1

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "eib")
        orig_last = ctx.shape[-1]
        edge_arr = np.array(self.edge_values, dtype=np.float32)
        concat_nodes, concat_inits, concat_out = _concat_edge_with_input(
            input_name, p, edge_arr, list(ctx.shape)
        )
        rhs = np.array(self.rhs_value, dtype=np.float32)
        op_out = f"{p}_op"
        op_node = helper.make_node(
            self.onnx_op, [concat_out, f"{p}_rhs"], [op_out]
        )
        slice_nodes, slice_inits, final_out = _slice_first_original(
            op_out, p, orig_last
        )
        inits = concat_inits + [numpy_helper.from_array(rhs, f"{p}_rhs")] + slice_inits
        return PatternInstance(
            self.name,
            self.category,
            concat_nodes + [op_node] + slice_nodes,
            inits,
            input_name,
            final_out,
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------


def _make_unary(*, tag: str, op: str,
                edges: Tuple[float, ...],
                op_attrs: Optional[dict] = None) -> type:
    name = f"ei_{op.lower()}_{tag}"
    return type(
        f"EdgeUnary_{op}_{tag}",
        (_UnaryEdgePattern,),
        {
            "name": name,
            "onnx_op": op,
            "edge_values": edges,
            "op_attrs": op_attrs or {},
            "target_optimization": f"edge_input_{op.lower()}",
        },
    )


def _make_binary(*, tag: str, op: str,
                 edges: Tuple[float, ...], rhs: float) -> type:
    name = f"ei_{op.lower()}_{tag}"
    return type(
        f"EdgeBinary_{op}_{tag}",
        (_BinaryEdgePattern,),
        {
            "name": name,
            "onnx_op": op,
            "edge_values": edges,
            "rhs_value": rhs,
            "target_optimization": f"edge_input_{op.lower()}",
        },
    )


_NAN = float("nan")
_POS_INF = float("inf")
_NEG_INF = float("-inf")
_NEG_ZERO = -0.0
_SUBNORMAL = 1e-40  # below fp32 normal range (~1.18e-38)

_UNARY_COMBOS: List[type] = [
    # Relu / Exp / Log / Sqrt / Reciprocal / Neg — NaN / 0 / -0 / Inf / subnormal
    _make_unary(tag="nan",     op="Relu",        edges=(_NAN,)),
    _make_unary(tag="neg_inf", op="Relu",        edges=(_NEG_INF,)),
    _make_unary(tag="nan",     op="Exp",         edges=(_NAN,)),
    _make_unary(tag="pos_inf", op="Exp",         edges=(_POS_INF,)),
    _make_unary(tag="neg_inf", op="Exp",         edges=(_NEG_INF,)),
    _make_unary(tag="zero",    op="Log",         edges=(0.0,)),
    _make_unary(tag="neg",     op="Log",         edges=(-1.0,)),
    _make_unary(tag="nan",     op="Log",         edges=(_NAN,)),
    _make_unary(tag="neg_zero", op="Sqrt",       edges=(_NEG_ZERO,)),
    _make_unary(tag="neg",      op="Sqrt",       edges=(-1.0,)),
    _make_unary(tag="zero",     op="Reciprocal", edges=(0.0,)),
    _make_unary(tag="neg_zero", op="Reciprocal", edges=(_NEG_ZERO,)),
    _make_unary(tag="nan",      op="Neg",        edges=(_NAN,)),
    _make_unary(tag="subnormal", op="Sigmoid",   edges=(_SUBNORMAL, -_SUBNORMAL)),
    _make_unary(tag="subnormal", op="Tanh",      edges=(_SUBNORMAL, -_SUBNORMAL)),
    # Softmax along last axis, with -inf and 0 logits.
    _make_unary(
        tag="neg_inf",
        op="Softmax",
        edges=(_NEG_INF, 0.0),
        op_attrs={"axis": -1},
    ),
]

_BINARY_COMBOS: List[type] = [
    # Div: x / 0 where x ≠ 0 vs x = 0
    _make_binary(tag="x_div_zero", op="Div",  edges=(1.0, -1.0), rhs=0.0),
    _make_binary(tag="zero_div_zero", op="Div", edges=(0.0, -0.0), rhs=0.0),
    # Pow(0, 0) → 1 per spec; some compilers disagree.
    _make_binary(tag="zero_pow_zero", op="Pow", edges=(0.0, 0.0), rhs=0.0),
    _make_binary(tag="zero_pow_neg",  op="Pow", edges=(0.0,), rhs=-1.0),
    # Min / Max vs NaN
    _make_binary(tag="nan_vs_1", op="Min", edges=(_NAN,), rhs=1.0),
    _make_binary(tag="nan_vs_1", op="Max", edges=(_NAN,), rhs=1.0),
    _make_binary(tag="inf_vs_1", op="Min", edges=(_POS_INF,), rhs=1.0),
    _make_binary(tag="neginf_vs_1", op="Max", edges=(_NEG_INF,), rhs=1.0),
    # Subnormal multiply / add — flush-to-zero behaviour
    _make_binary(tag="subnormal_mul_1", op="Mul",
                 edges=(_SUBNORMAL, _SUBNORMAL), rhs=1.0),
    _make_binary(tag="subnormal_add_0", op="Add",
                 edges=(_SUBNORMAL, -_SUBNORMAL), rhs=0.0),
]


# ---------------------------------------------------------------------------
# ReduceLogSumExp specific pattern (OV #21988)
# ---------------------------------------------------------------------------


class ReduceLogSumExpOverflow(OTP):
    """
    Classic ``ReduceLogSumExp`` overflow: input magnitudes ≥ 88.7 cause
    ``exp(x)`` to overflow fp32, so the reduction silently returns ``+Inf``
    unless the compiler applies the max-subtraction trick.
    The pattern concats large-magnitude edge values onto the input along the
    last axis, reduces along that axis, then broadcasts the scalar result back
    to the original shape.
    """

    name = "ei_reducelogsumexp_overflow"
    category = CAT_BROADCAST
    target_optimization = "reducelogsumexp_overflow"

    def is_compatible(self, ctx: StructuralContext) -> bool:
        return ctx.dtype == "float32" and ctx.rank >= 1

    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        p = self._p(node_id, "ei_rlse")
        edge_arr = np.array([100.0, 88.8, 50.0], dtype=np.float32)
        concat_nodes, concat_inits, concat_out = _concat_edge_with_input(
            input_name, p, edge_arr, list(ctx.shape)
        )
        # Reduce along last axis, keeping dims so Add is broadcastable.
        # Opset 18+: ``axes`` is an input tensor, not an attribute.
        red_out = f"{p}_rlse"
        axes_np = np.array([-1], dtype=np.int64)
        nodes = concat_nodes + [
            helper.make_node(
                "ReduceLogSumExp",
                [concat_out, f"{p}_axes"], [red_out],
                keepdims=1,
            ),
            # Scale by epsilon so the reduction's value contaminates the input
            # (preserves the input shape with broadcasting).
            helper.make_node("Mul", [red_out, f"{p}_eps"], [f"{p}_scaled"]),
            helper.make_node("Add", [input_name, f"{p}_scaled"], [f"{p}_out"]),
        ]
        inits = concat_inits + [
            numpy_helper.from_array(axes_np, f"{p}_axes"),
            numpy_helper.from_array(np.array(1e-6, dtype=np.float32), f"{p}_eps"),
        ]
        return PatternInstance(
            self.name,
            self.category,
            nodes,
            inits,
            input_name,
            f"{p}_out",
            StructuralContext(ctx.rank, list(ctx.shape), ctx.dtype, ctx.layout),
        )


# ---------------------------------------------------------------------------
# Public surfaces
# ---------------------------------------------------------------------------

_ALL_CLASSES: List[type] = _UNARY_COMBOS + _BINARY_COMBOS + [ReduceLogSumExpOverflow]

ALL_EDGE_INPUT_PATTERNS: List[OTP] = [cls() for cls in _ALL_CLASSES]

EDGE_INPUT_BY_CATEGORY: dict[str, List[OTP]] = {
    CAT_BROADCAST: ALL_EDGE_INPUT_PATTERNS,
}
