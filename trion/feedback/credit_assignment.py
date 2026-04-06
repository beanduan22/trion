"""
Pattern-Level Credit Assignment and Feedback Update (Section 2.3).

U(p) = R(p)/(N(p)+ε) + λ/√(N(p)+1)
U(g) = R(g)/(N(g)+ε) + λ/√(N(g)+1)

Policies are soft-max'd over admissible sets in PatternAwareSearchSpace.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple

from ..config import TrionConfig


class CreditAssignment:
    """
    Maintains cumulative rewards R(p)/R(g) and counts N(p)/N(g),
    and computes UCB-style utility scores used to update the sampling policies.
    """

    def __init__(self, config: TrionConfig, categories: List[str]) -> None:
        self.lam = config.exploration_coefficient
        self.eps = config.epsilon

        self._cat_R: Dict[str, float] = {c: 0.0 for c in categories}
        self._cat_N: Dict[str, int]   = {c: 0   for c in categories}

        self._pat_R: Dict[str, Dict[str, float]] = {}
        self._pat_N: Dict[str, Dict[str, int]]   = {}

    def register_patterns(self, category: str, pattern_names: List[str]) -> None:
        """Call once per (category, pattern) at startup."""
        if category not in self._pat_R:
            self._pat_R[category] = {}
            self._pat_N[category] = {}
        for name in pattern_names:
            self._pat_R[category].setdefault(name, 0.0)
            self._pat_N[category].setdefault(name, 0)

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        pattern_sequence: List[Tuple[str, str]],  # [(cat, pat_name), ...]
        discrepancy_score: float,
    ) -> None:
        """
        Distribute model-level score uniformly to constituent patterns.
        r_t = (1/K) * S(m)
        """
        K = len(pattern_sequence)
        if K == 0:
            return
        per_pattern_reward = discrepancy_score / K

        for cat, pat_name in pattern_sequence:
            # Category
            self._cat_R[cat] += per_pattern_reward
            self._cat_N[cat] += 1
            # Pattern
            self._pat_R[cat][pat_name] = (
                self._pat_R[cat].get(pat_name, 0.0) + per_pattern_reward
            )
            self._pat_N[cat][pat_name] = (
                self._pat_N[cat].get(pat_name, 0) + 1
            )

    def update_crash(
        self,
        pattern_sequence: List[Tuple[str, str]],
        crash_type: str,
    ) -> None:
        """
        Update credit accounting for a model that only produced crashes.

        Frontend crashes (onnx2torch/tracing failures): no update at all —
        the crash is not the compiler's fault and should not influence pattern selection.

        Backend crashes: apply a small negative reward so that patterns repeatedly
        causing real backend failures are gently deprioritized.
        """
        if crash_type == "frontend":
            return
        K = len(pattern_sequence)
        if K == 0:
            return
        penalty = -0.05 / K
        for cat, pat_name in pattern_sequence:
            self._cat_R[cat] += penalty
            self._cat_N[cat] += 1
            self._pat_R[cat][pat_name] = (
                self._pat_R[cat].get(pat_name, 0.0) + penalty
            )
            self._pat_N[cat][pat_name] = (
                self._pat_N[cat].get(pat_name, 0) + 1
            )

    # ── Utility computation ───────────────────────────────────────────────────

    def _ucb(self, total_reward: float, count: int) -> float:
        return (total_reward / (count + self.eps)
                + self.lam / math.sqrt(count + 1))

    def category_utilities(self) -> Dict[str, float]:
        return {
            cat: self._ucb(self._cat_R[cat], self._cat_N[cat])
            for cat in self._cat_R
        }

    def pattern_utilities(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for cat in self._pat_R:
            result[cat] = {
                pn: self._ucb(self._pat_R[cat][pn], self._pat_N[cat][pn])
                for pn in self._pat_R[cat]
            }
        return result

    # ── Stats summary ─────────────────────────────────────────────────────────

    def all_pattern_usage(self) -> List[Tuple[str, str, float, int]]:
        """Return every registered pattern with (cat, name, avg_reward, count),
        sorted by category then name. Used for full-coverage reporting."""
        rows = []
        for cat in sorted(self._pat_R):
            for pn in sorted(self._pat_R[cat]):
                cnt = self._pat_N[cat][pn]
                avg = self._pat_R[cat][pn] / (cnt + self.eps)
                rows.append((cat, pn, avg, cnt))
        return rows

    def top_patterns(self, n: int = 10) -> List[Tuple[str, str, float, int]]:
        """Return top-n patterns by average reward."""
        rows = []
        for cat in self._pat_R:
            for pn in self._pat_R[cat]:
                cnt = self._pat_N[cat][pn]
                avg = self._pat_R[cat][pn] / (cnt + self.eps)
                rows.append((cat, pn, avg, cnt))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:n]

    def summary(self) -> str:
        lines = ["=== Credit Assignment Summary ==="]
        cat_u = self.category_utilities()
        for cat, u in sorted(cat_u.items(), key=lambda x: -x[1]):
            lines.append(
                f"  {cat:20s}  U={u:.4f}  N={self._cat_N[cat]:5d}"
                f"  R={self._cat_R[cat]:.4f}"
            )
        lines.append("")
        lines.append("Top-10 patterns by avg reward:")
        for cat, pn, avg, cnt in self.top_patterns(10):
            lines.append(f"  [{cat}] {pn:35s}  avg={avg:.4f}  N={cnt:4d}")
        return "\n".join(lines)
