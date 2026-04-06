"""
Input Generation and Mutation (Section 2.4).

Three mutation strategies (all numerically stable):
  1. Additive noise     — small Gaussian perturbations
  2. Scale variation    — mild scale changes (0.1–2.0×)
  3. Sparse activation  — most values zeroed out
  4. Boundary values    — finite edge-case values (0, ±1, ±0.5)
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


class InputMutator:
    """
    Generates a base input and applies lightweight, numerically stable mutations.
    Extreme values (near-inf, ×1000, ×1e-5) are intentionally excluded to avoid
    false-positive divergences caused by floating-point overflow/underflow.
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
        Return a list of *n* mutated inputs using stable strategies only.
        """
        strategies = [
            self._additive_noise,
            self._scale_variation,
            self._sparse_activation,
            self._boundary_values,
        ]
        results = []
        for i in range(n):
            fn = strategies[i % len(strategies)]
            results.append(fn(x))
        return results

    # ── Mutation strategies ───────────────────────────────────────────────────

    def _additive_noise(self, x: np.ndarray) -> np.ndarray:
        """Small Gaussian perturbation (1% of input std)."""
        std = float(np.std(x)) * 0.01 + 1e-7
        noise = self.rng.normal(0, std, x.shape).astype(x.dtype)
        return x + noise

    def _scale_variation(self, x: np.ndarray) -> np.ndarray:
        """Mild global scale — stays within representable float32 range."""
        scale = float(self.rng.choice([0.1, 0.5, 1.0, 2.0]))
        return (x * scale).astype(x.dtype)

    def _sparse_activation(self, x: np.ndarray) -> np.ndarray:
        """Set 90% of values to 0 — mimics sparse activations."""
        mask = self.rng.random(x.shape) > 0.9
        return (x * mask).astype(x.dtype)

    def _boundary_values(self, x: np.ndarray) -> np.ndarray:
        """Inject finite edge-case values (no inf/nan/overflow triggers)."""
        x2 = x.copy()
        flat = x2.ravel()
        n_inject = max(1, len(flat) // 20)
        idx = self.rng.choice(len(flat), n_inject, replace=False)
        vals = self.rng.choice(
            [0.0, -0.0, 1.0, -1.0, 0.5, -0.5],
            n_inject,
        )
        flat[idx] = vals
        return x2

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
