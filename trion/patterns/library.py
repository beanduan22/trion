"""
OTPLibrary: aggregates all patterns and exposes category-level access.
"""
from __future__ import annotations
from typing import Dict, List

from .base import OTP, ALL_CATEGORIES
from .fusion_patterns import ALL_FUSION_PATTERNS
from .layout_patterns import ALL_LAYOUT_PATTERNS
from .broadcast_patterns import ALL_BROADCAST_PATTERNS
from .normalization_patterns import ALL_NORMALIZATION_PATTERNS
from .branch_patterns import ALL_BRANCH_PATTERNS
from .constant_patterns import ALL_CONSTANT_PATTERNS
from .attention_patterns import ALL_ATTENTION_PATTERNS


class OTPLibrary:
    """
    𝒫 — the full OTP library.
    Provides category-level and flat access to all patterns.
    """

    _category_map: Dict[str, List[OTP]] = {
        "fusion":        ALL_FUSION_PATTERNS,
        "layout":        ALL_LAYOUT_PATTERNS,
        "broadcast":     ALL_BROADCAST_PATTERNS,
        "normalization": ALL_NORMALIZATION_PATTERNS,
        "branch":        ALL_BRANCH_PATTERNS,
        "constant":      ALL_CONSTANT_PATTERNS,
        "attention":     ALL_ATTENTION_PATTERNS,
    }

    def __init__(self) -> None:
        self._map: Dict[str, Dict[str, OTP]] = {}
        for cat, patterns in self._category_map.items():
            self._map[cat] = {p.name: p for p in patterns}

    # ── Query helpers ─────────────────────────────────────────────────────────

    @property
    def categories(self) -> List[str]:
        return list(self._map.keys())

    def patterns_in(self, category: str) -> List[OTP]:
        return list(self._map[category].values())

    def all_patterns(self) -> List[OTP]:
        result = []
        for patterns in self._map.values():
            result.extend(patterns.values())
        return result

    def get(self, category: str, name: str) -> OTP:
        return self._map[category][name]

    def summary(self) -> str:
        lines = [f"OTPLibrary — {len(self.all_patterns())} patterns total"]
        for cat in self.categories:
            names = [p.name for p in self.patterns_in(cat)]
            lines.append(f"  [{cat}] ({len(names)}): {', '.join(names)}")
        return "\n".join(lines)
