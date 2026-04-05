"""
Base classes for Optimization Trigger Patterns (OTP).
p = (G_p, I_p, O_p, C_p, T_p)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..generation.context import StructuralContext


@dataclass
class PatternInstance:
    """
    A concrete instantiation p̂ = Instantiate(p, θ).
    Holds the ONNX subgraph nodes and initializers, plus the wiring info
    needed by the composer.
    """
    pattern_name: str
    category: str
    nodes: List[onnx.NodeProto]
    initializers: List[onnx.TensorProto]
    # The name of the tensor this subgraph reads as primary input.
    input_name: str
    # The name of the output tensor produced by this subgraph.
    output_name: str
    output_context: StructuralContext
    attributes: Dict[str, Any] = field(default_factory=dict)


class OTP(ABC):
    """
    Abstract base class for all optimization trigger patterns.

    Subclasses must declare:
        name            : str
        category        : str  (one of the 6 category constants below)
        target_optimization : str
    """
    name: str = ""
    category: str = ""
    target_optimization: str = ""

    # ── Public interface ──────────────────────────────────────────────────────

    @abstractmethod
    def is_compatible(self, ctx: StructuralContext) -> bool:
        """Return True iff I_p is compatible with *ctx*."""

    @abstractmethod
    def instantiate(
        self,
        input_name: str,
        ctx: StructuralContext,
        rng: np.random.Generator,
        node_id: int,
    ) -> Optional[PatternInstance]:
        """
        Sample θ ∈ Θ(p) and build the ONNX subgraph.
        Return None if C_p cannot be satisfied for *ctx*.
        """

    # ── Helper utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _p(node_id: int, tag: str) -> str:
        """Generate a unique tensor/node name."""
        return f"n{node_id}_{tag}"

    @staticmethod
    def _const(name: str, array: np.ndarray) -> onnx.TensorProto:
        return numpy_helper.from_array(array.astype(array.dtype), name=name)

    @staticmethod
    def _rand_channels(rng: np.random.Generator, candidates=(32, 64, 128, 256)) -> int:
        return int(rng.choice(candidates))

    @staticmethod
    def _make_conv_weight(
        rng: np.random.Generator,
        out_c: int,
        in_c: int,
        k: int,
    ) -> np.ndarray:
        fan_in = in_c * k * k
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(0, std, (out_c, in_c, k, k)).astype(np.float32)


# ── Category constants ─────────────────────────────────────────────────────
CAT_FUSION = "fusion"
CAT_LAYOUT = "layout"
CAT_BROADCAST = "broadcast"
CAT_NORMALIZATION = "normalization"
CAT_BRANCH = "branch"
CAT_CONSTANT = "constant"

ALL_CATEGORIES = [
    CAT_FUSION,
    CAT_LAYOUT,
    CAT_BROADCAST,
    CAT_NORMALIZATION,
    CAT_BRANCH,
    CAT_CONSTANT,
]
