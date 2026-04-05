from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class StructuralContext:
    """
    c_t = (rank, shape, dtype, layout)
    Tracks the structural properties of the tensor flowing between patterns.
    """
    rank: int
    shape: List[int]
    dtype: str = "float32"    # float32 | float16 | int32 | int64
    layout: str = "NCHW"      # NCHW | NHWC | NC | NLC | ND

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def shape_category(self) -> str:
        if self.rank == 4:
            return "spatial_4d"
        if self.rank == 3:
            return "spatial_3d"
        if self.rank == 2:
            return "matrix"
        if self.rank == 1:
            return "vector"
        return "nd"

    def is_spatial(self) -> bool:
        return self.rank >= 3

    def is_matrix(self) -> bool:
        return self.rank == 2

    def batch(self) -> int:
        return self.shape[0]

    def channels(self) -> Optional[int]:
        if self.rank < 2:
            return None
        if self.layout in ("NCHW", "NC"):
            return self.shape[1]
        if self.layout == "NHWC":
            return self.shape[-1]
        if self.layout == "NLC":
            return self.shape[-1]
        return self.shape[1]

    def spatial_dims(self) -> Optional[List[int]]:
        if self.rank == 4:
            if self.layout == "NCHW":
                return list(self.shape[2:])
            if self.layout == "NHWC":
                return list(self.shape[1:3])
        return None

    def num_elements(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    def to_onnx_dtype(self) -> int:
        import onnx
        _map = {
            "float32": onnx.TensorProto.FLOAT,
            "float16": onnx.TensorProto.FLOAT16,
            "int32":   onnx.TensorProto.INT32,
            "int64":   onnx.TensorProto.INT64,
        }
        return _map.get(self.dtype, onnx.TensorProto.FLOAT)

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def copy(self) -> "StructuralContext":
        return StructuralContext(
            rank=self.rank,
            shape=list(self.shape),
            dtype=self.dtype,
            layout=self.layout,
        )
