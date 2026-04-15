"""
OTPLibrary: aggregates all patterns and exposes category-level access.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from .base import OTP, ALL_CATEGORIES
from .fusion_patterns import ALL_FUSION_PATTERNS
from .layout_patterns import ALL_LAYOUT_PATTERNS
from .broadcast_patterns import ALL_BROADCAST_PATTERNS
from .normalization_patterns import ALL_NORMALIZATION_PATTERNS
from .branch_patterns import ALL_BRANCH_PATTERNS
from .constant_patterns import ALL_CONSTANT_PATTERNS
from .attention_patterns import ALL_ATTENTION_PATTERNS
from .issue_mined_patterns import ISSUE_MINED_BY_CATEGORY
from .zoo_mined_patterns import ZOO_MINED_BY_CATEGORY
from .op_coverage_patterns import OP_COVERAGE_BY_CATEGORY
from .op_coverage_v2 import OP_COVERAGE_V2_BY_CATEGORY
from .op_coverage_v3 import OP_COVERAGE_V3_BY_CATEGORY
from .issue_mined_v2 import ISSUE_MINED_V2_BY_CATEGORY
from .op_coverage_v4 import OP_COVERAGE_V4_BY_CATEGORY
from .op_coverage_v5 import OP_COVERAGE_V5_BY_CATEGORY
from .op_coverage_v6 import OP_COVERAGE_V6_BY_CATEGORY
from .universal_patterns import UNIVERSAL_BY_CATEGORY


class OTPLibrary:
    """
    𝒫 — the full OTP library.
    Provides category-level and flat access to all patterns.
    """

    @staticmethod
    def _merge(*groups):
        out: List[OTP] = []
        for g in groups:
            out.extend(g)
        return out

    @staticmethod
    def _cat(category: str) -> List[OTP]:
        base_maps = {
            "fusion": ALL_FUSION_PATTERNS,
            "layout": ALL_LAYOUT_PATTERNS,
            "broadcast": ALL_BROADCAST_PATTERNS,
            "normalization": ALL_NORMALIZATION_PATTERNS,
            "branch": ALL_BRANCH_PATTERNS,
            "constant": ALL_CONSTANT_PATTERNS,
            "attention": ALL_ATTENTION_PATTERNS,
        }
        extra_sources = [
            ISSUE_MINED_BY_CATEGORY,
            ZOO_MINED_BY_CATEGORY,
            OP_COVERAGE_BY_CATEGORY,
            OP_COVERAGE_V2_BY_CATEGORY,
            OP_COVERAGE_V3_BY_CATEGORY,
            ISSUE_MINED_V2_BY_CATEGORY,
            OP_COVERAGE_V4_BY_CATEGORY,
            OP_COVERAGE_V5_BY_CATEGORY,
            OP_COVERAGE_V6_BY_CATEGORY,
            UNIVERSAL_BY_CATEGORY,
        ]
        out: List[OTP] = list(base_maps.get(category, []))
        for src in extra_sources:
            out.extend(src.get(category, []))
        return out

    _category_map: Dict[str, List[OTP]] = {}

    def __init__(
        self,
        compat_json: Optional[str] = None,
        active_backends: Optional[List[str]] = None,
    ) -> None:
        """
        Build the library, optionally filtered by a pattern-compatibility
        cache produced by `tools/check_pattern_compat.py`.

        A pattern is included only if every `active_backends` entry is in
        its supported-backend set in the cache.  When `compat_json` is not
        given (or the file does not exist) all patterns are included.
        """
        self._map: Dict[str, Dict[str, OTP]] = {}

        allowed: Optional[Set[str]] = None
        if compat_json and active_backends and os.path.exists(compat_json):
            try:
                with open(compat_json) as f:
                    cache = json.load(f)
                cache_backends = set(cache.get("backends", []))
                missing = set(active_backends) - cache_backends
                if missing:
                    logger.warning(
                        "Compat cache at %s does not cover these backends: %s "
                        "— ignoring cache and using full library.",
                        compat_json, sorted(missing),
                    )
                else:
                    def _is_pass(per_backend: dict, bname: str) -> bool:
                        """A pattern passes on `bname` iff the cache entry
                        reports status=='pass' (new schema) OR the value is
                        a truthy boolean (old boolean schema). Anything else
                        — crash, diverge, missing — is a no."""
                        entry = per_backend.get(bname)
                        if isinstance(entry, dict):
                            return entry.get("status") == "pass"
                        return bool(entry)
                    allowed = {
                        name
                        for name, per_backend in cache.get("patterns", {}).items()
                        if all(_is_pass(per_backend, b) for b in active_backends)
                    }
                    total_cached = len(cache.get("patterns", {}))
                    # Count how many patterns are specifically excluded because
                    # they diverge (not crash) on at least one backend — that's
                    # the number of patterns that would have caused a
                    # pattern-level FP if left in the campaign.
                    diverge_count = sum(
                        1
                        for per_backend in cache.get("patterns", {}).values()
                        if any(
                            isinstance(per_backend.get(b), dict)
                            and per_backend.get(b, {}).get("status") == "diverge"
                            for b in active_backends
                        )
                    )
                    logger.info(
                        "Pattern compat filter: %d/%d patterns pass on all of %s "
                        "(%d excluded because they diverge on at least one backend — "
                        "these would have been pattern-level false positives).",
                        len(allowed), total_cached, active_backends, diverge_count,
                    )
            except Exception as exc:
                logger.warning("Failed to read compat cache %s: %s", compat_json, exc)

        for cat in ("fusion", "layout", "broadcast", "normalization",
                    "branch", "constant", "attention"):
            patterns = self._cat(cat)
            if allowed is not None:
                patterns = [p for p in patterns if p.name in allowed]
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
