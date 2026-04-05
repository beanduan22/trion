"""
Input Generation and Mutation (Section 2.4).

Three mutation strategies:
  1. Additive noise     — small perturbations
  2. Scale variation    — test numerical stability
  3. Boundary values    — trigger edge-case behaviors
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


class InputMutator:
    """
    Generates a base input and applies lightweight mutations.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_base(
        self,
        shape: List[int],
        dtype: str = "float32",
        distribution: str = "normal",
    ) -> np.ndarray:
        """Sample the base input from a standard distribution."""
        if distribution == "normal":
            x = self.rng.standard_normal(shape).astype(dtype)
        elif distribution == "uniform":
            x = self.rng.uniform(-1, 1, shape).astype(dtype)
        else:
            x = self.rng.standard_normal(shape).astype(dtype)
        return x

    def mutate(self, x: np.ndarray, n: int = 3) -> List[np.ndarray]:
        """
        Return a list of *n* mutated inputs.
        Each mutation is drawn from a different strategy.
        """
        strategies = [
            self._additive_noise,
            self._scale_variation,
            self._boundary_values,
            self._sparse_activation,
            self._large_values,
            self._near_zero,
        ]
        results = []
        for i in range(n):
            fn = strategies[i % len(strategies)]
            results.append(fn(x))
        return results

    # ── Mutation strategies ───────────────────────────────────────────────────

    def _additive_noise(self, x: np.ndarray) -> np.ndarray:
        """Small Gaussian perturbation."""
        std = float(np.std(x)) * 0.01 + 1e-7
        noise = self.rng.normal(0, std, x.shape).astype(x.dtype)
        return x + noise

    def _scale_variation(self, x: np.ndarray) -> np.ndarray:
        """Random global scale (tests numerical stability)."""
        scale = float(self.rng.choice([1e-3, 1e-2, 0.1, 1.0, 10.0, 1e3]))
        return (x * scale).astype(x.dtype)

    def _boundary_values(self, x: np.ndarray) -> np.ndarray:
        """Inject a fraction of near-inf / near-zero values."""
        x2 = x.copy()
        flat = x2.ravel()
        n_inject = max(1, len(flat) // 20)
        idx = self.rng.choice(len(flat), n_inject, replace=False)
        vals = self.rng.choice(
            [np.finfo(np.float32).max * 0.5,
             np.finfo(np.float32).tiny,
             0.0, -0.0,
             1.0, -1.0],
            n_inject,
        )
        flat[idx] = vals
        return x2

    def _sparse_activation(self, x: np.ndarray) -> np.ndarray:
        """Set most values to 0 — mimics sparse activations."""
        mask = self.rng.random(x.shape) > 0.9
        return (x * mask).astype(x.dtype)

    def _large_values(self, x: np.ndarray) -> np.ndarray:
        """Scale up uniformly — can trigger overflow in FP16."""
        return (x * 100.0).astype(x.dtype)

    def _near_zero(self, x: np.ndarray) -> np.ndarray:
        """Shift values very close to zero — exercises division stability."""
        return (x * 1e-5).astype(x.dtype)

    # ── Convenience: all inputs for one model ────────────────────────────────

    def generate_all(
        self,
        shape: List[int],
        dtype: str = "float32",
        num_mutations: int = 3,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Returns base + mutations as a list of feed-dicts.
        The input tensor name is fixed to 'model_input' to match the composer.
        """
        base = self.generate_base(shape, dtype)
        variants = [base] + self.mutate(base, num_mutations)
        return [{"model_input": v} for v in variants]
