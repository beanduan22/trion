"""
Abstract backend interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import onnx


class BackendResult:
    """Holds the execution result from one backend."""
    def __init__(
        self,
        output: Optional[np.ndarray],
        error: Optional[str] = None,
        crashed: bool = False,
    ) -> None:
        self.output  = output
        self.error   = error
        self.crashed = crashed

    @property
    def ok(self) -> bool:
        return not self.crashed and self.output is not None


class BackendBase(ABC):
    """Common interface for all DL compiler backends."""
    name: str = ""

    @abstractmethod
    def run(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        optimized: bool = True,
    ) -> BackendResult:
        """
        Execute *model* with *inputs*.
        If the backend supports it, *optimized=False* disables optimizations.
        """

    def is_available(self) -> bool:
        """Return False if required packages are not installed."""
        return True
