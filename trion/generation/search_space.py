"""
Pattern-Aware Search Space (Section 2.2 of the paper).

Implements hierarchical model generation:
  1. Sample category  g_t ~ π_cat(· | c_t)
  2. Sample pattern   p_t ~ π_pat(· | g_t, c_t)
  3. Instantiate      p̂_t = Instantiate(p_t, θ_t)
  4. Update context   c_{t+1} = CtxOut(p̂_t)
"""
from __future__ import annotations
import io
import logging
from typing import List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, TensorProto

from ..config import TrionConfig
from ..generation.context import StructuralContext
from ..patterns.base import OTP, PatternInstance
from ..patterns.library import OTPLibrary

logger = logging.getLogger(__name__)


class GeneratedModel:
    """Wraps the composed ONNX model together with the pattern sequence used."""
    def __init__(
        self,
        onnx_model: onnx.ModelProto,
        pattern_sequence: List[Tuple[str, str]],   # [(category, name), ...]
        input_shape: List[int],
        input_dtype: str,
    ) -> None:
        self.onnx_model = onnx_model
        self.pattern_sequence = pattern_sequence
        self.input_shape = input_shape
        self.input_dtype = input_dtype


class PatternAwareSearchSpace:
    """
    Hierarchical generator for candidate test models.

    Policies π_cat and π_pat start uniform and are updated externally
    via `update_policies(cat_utils, pat_utils)`.
    """

    def __init__(
        self,
        library: OTPLibrary,
        config: TrionConfig,
        rng: np.random.Generator,
    ) -> None:
        self.lib    = library
        self.config = config
        self.rng    = rng

        # Softmax-normalised weights (updated by credit assignment)
        self._cat_logits: dict[str, float] = {c: 0.0 for c in library.categories}
        self._pat_logits: dict[str, dict[str, float]] = {
            cat: {p.name: 0.0 for p in library.patterns_in(cat)}
            for cat in library.categories
        }

    # ── Policy update ─────────────────────────────────────────────────────────

    def update_policies(
        self,
        cat_utilities: dict[str, float],
        pat_utilities: dict[str, dict[str, float]],
    ) -> None:
        """Called by CreditAssignment after each model evaluation."""
        self._cat_logits.update(cat_utilities)
        for cat, pu in pat_utilities.items():
            if cat in self._pat_logits:
                self._pat_logits[cat].update(pu)

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def _softmax_sample(self, logits: dict[str, float], temperature: float = 1.0) -> str:
        """Sample from a softmax distribution over UCB utility logits.

        UCB values cluster near λ (≈1.0) so raw logits have tiny spread.
        Z-score normalise first so relative UCB differences drive selection,
        then apply temperature for residual exploration control.
        temperature < 1 concentrates on high-utility items;
        temperature = 1 is standard softmax over z-scored utilities.
        """
        keys = list(logits.keys())
        if len(keys) == 1:
            return keys[0]
        vals = np.array([logits[k] for k in keys], dtype=float)
        std = vals.std()
        if std > 1e-9:
            vals = (vals - vals.mean()) / std   # z-score: full spread regardless of scale
        else:
            vals = vals - vals.mean()           # all equal → uniform after softmax
        vals = vals / max(temperature, 1e-6)
        vals -= vals.max()                      # numerical stability
        probs = np.exp(vals)
        probs /= probs.sum()
        return keys[self.rng.choice(len(keys), p=probs)]

    def _admissible_categories(self, ctx: StructuralContext) -> List[str]:
        return [
            cat for cat in self.lib.categories
            if any(p.is_compatible(ctx) for p in self.lib.patterns_in(cat))
        ]

    def _admissible_patterns(self, cat: str, ctx: StructuralContext) -> List[OTP]:
        return [p for p in self.lib.patterns_in(cat) if p.is_compatible(ctx)]

    # ── Model generation ──────────────────────────────────────────────────────

    def generate(self) -> Optional[GeneratedModel]:
        """
        Build one candidate model by composing K pattern instances.
        Returns None if composition fails (all retries exhausted).
        """
        cfg = self.config
        ctx = self._make_seed_context()
        input_shape = list(ctx.shape)
        input_dtype = ctx.dtype

        all_nodes:       list = []
        all_initializers: list = []
        pattern_sequence: list[tuple[str, str]] = []
        current_name = "model_input"

        for step in range(cfg.pattern_budget):
            adm_cats = self._admissible_categories(ctx)
            if not adm_cats:
                break

            strategy = getattr(cfg, "sampling_strategy", "ucb")

            if strategy == "uniform":
                # ── Uniform random category + pattern (broad coverage) ──
                cat = adm_cats[self.rng.integers(0, len(adm_cats))]
                adm_pats = self._admissible_patterns(cat, ctx)
                if not adm_pats:
                    continue
                pat = adm_pats[self.rng.integers(0, len(adm_pats))]
            else:
                # ── UCB-driven softmax sampling ──
                cat_logits = {c: self._cat_logits[c] for c in adm_cats}
                cat = self._softmax_sample(cat_logits)

                adm_pats = self._admissible_patterns(cat, ctx)
                if not adm_pats:
                    continue
                pat_logits = {p.name: self._pat_logits[cat][p.name] for p in adm_pats}
                pat_name = self._softmax_sample(pat_logits)
                pat = next(p for p in adm_pats if p.name == pat_name)

            # ── 3. Instantiate ────────────────────────────────────────────
            instance: Optional[PatternInstance] = None
            for _retry in range(5):
                try:
                    instance = pat.instantiate(current_name, ctx, self.rng, step * 100 + _retry)
                    break
                except Exception as exc:
                    logger.debug("Instantiation error %s (retry %d): %s",
                                 pat.name, _retry, exc)

            if instance is None:
                continue

            all_nodes.extend(instance.nodes)
            all_initializers.extend(instance.initializers)
            pattern_sequence.append((cat, pat.name))

            # ── 4. Update context ─────────────────────────────────────────
            current_name = instance.output_name
            ctx = instance.output_context

        if not pattern_sequence:
            return None

        onnx_model = self._build_onnx_model(
            all_nodes, all_initializers,
            input_name="model_input",
            output_name=current_name,
            input_shape=input_shape,
            output_shape=list(ctx.shape),
            input_dtype=input_dtype,
        )
        if onnx_model is None:
            return None

        return GeneratedModel(onnx_model, pattern_sequence, input_shape, input_dtype)

    # ── Seed context ──────────────────────────────────────────────────────────

    def _make_seed_context(self) -> StructuralContext:
        cfg = self.config
        choice = self.rng.integers(0, 4)
        if choice == 0:
            # 4-D spatial (most patterns target this)
            return StructuralContext(
                rank=4,
                shape=[cfg.batch_size,
                       cfg.default_channels,
                       cfg.default_spatial_size,
                       cfg.default_spatial_size],
                dtype="float32",
                layout="NCHW",
            )
        elif choice == 1:
            # 2-D matrix
            feat = int(self.rng.choice([64, 128, 256]))
            return StructuralContext(
                rank=2,
                shape=[cfg.batch_size, feat],
                dtype="float32",
                layout="NC",
            )
        elif choice == 2:
            # 4-D with larger channel count (for SE block, group norm, etc.)
            c = int(self.rng.choice([32, 64, 128]))
            sp = int(self.rng.choice([16, 32]))
            return StructuralContext(
                rank=4,
                shape=[cfg.batch_size, c, sp, sp],
                dtype="float32",
                layout="NCHW",
            )
        else:
            # 3-D sequence [B, S, D] for attention / transformer patterns
            # D must be divisible by 8 for multi-head attention (4 or 8 heads)
            seq_len = int(self.rng.choice([16, 32, 64]))
            hidden  = int(self.rng.choice([64, 128, 256]))
            return StructuralContext(
                rank=3,
                shape=[cfg.batch_size, seq_len, hidden],
                dtype="float32",
                layout="NLC",
            )

    # ── ONNX model builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_onnx_model(
        nodes, initializers,
        input_name, output_name,
        input_shape, output_shape,
        input_dtype="float32",
    ) -> Optional[onnx.ModelProto]:
        """Assemble an ONNX ModelProto from nodes + initializers."""
        try:
            dtype_map = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}
            onnx_dtype = dtype_map.get(input_dtype, TensorProto.FLOAT)

            input_vi  = helper.make_tensor_value_info(input_name,  onnx_dtype, input_shape)
            # Output dtype must match the dtype that flows out of the final
            # pattern, not be hardcoded to FLOAT — otherwise non-fp32 graphs
            # produce ValueInfo type mismatches that backends report as bugs.
            output_vi = helper.make_tensor_value_info(output_name, onnx_dtype, output_shape)

            graph = helper.make_graph(
                nodes,
                "trion_graph",
                [input_vi],
                [output_vi],
                initializer=initializers,
            )
            model = helper.make_model(graph, opset_imports=[
                helper.make_opsetid("", 17),   # default domain, opset 17
            ])
            model.ir_version = 8
            onnx.checker.check_model(model)
            return model
        except Exception as exc:
            logger.debug("ONNX model build failed: %s", exc)
            return None
