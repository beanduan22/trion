"""
Trion Main Testing Loop.

for i in range(config.num_models):
    model  = search_space.generate()
    inputs = input_mutator.generate_all(...)
    for x in inputs:
        report = oracle.score(model, x, model_id=i)
    credit.update(pattern_sequence, max_score)
    search_space.update_policies(...)
"""
from __future__ import annotations
import json
import logging
import os
import time
from typing import Optional

import numpy as np
import onnx

from .config import TrionConfig
from .feedback.credit_assignment import CreditAssignment
from .generation.search_space import PatternAwareSearchSpace, GeneratedModel
from .mutation.input_mutator import InputMutator
from .oracle.oracle import DiscrepancyOracle, OracleReport
from .patterns.library import OTPLibrary

logger = logging.getLogger(__name__)


class TrionRunner:
    """
    Orchestrates the full Trion testing pipeline.

    Usage:
        runner = TrionRunner(config)
        runner.run()
    """

    def __init__(self, config: Optional[TrionConfig] = None) -> None:
        self.config = config or TrionConfig()
        cfg = self.config

        logging.basicConfig(
            level=logging.INFO if cfg.verbose else logging.WARNING,
            format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        )

        rng = np.random.default_rng(cfg.seed)

        self.library     = OTPLibrary()
        self.search_space = PatternAwareSearchSpace(self.library, cfg, rng)
        self.mutator      = InputMutator(rng)
        self.oracle       = DiscrepancyOracle(cfg)

        self.credit = CreditAssignment(cfg, self.library.categories)
        for cat in self.library.categories:
            self.credit.register_patterns(
                cat, [p.name for p in self.library.patterns_in(cat)]
            )

        # Statistics
        self._n_generated  = 0
        self._n_bugs       = 0
        self._n_crashes    = 0
        self._total_score  = 0.0

        if cfg.save_bugs:
            os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Public entry-point ────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self.config
        logger.info("=== Trion ===")
        logger.info(self.library.summary())
        logger.info(self.oracle.summary())
        logger.info("Starting %d model generations …", cfg.num_models)

        t0 = time.time()
        for i in range(cfg.num_models):
            self._run_one(i)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "[%5d/%d]  bugs=%d  crashes=%d  avg_score=%.4f  "
                    "elapsed=%.1fs",
                    i + 1, cfg.num_models,
                    self._n_bugs, self._n_crashes,
                    self._total_score / max(self._n_generated, 1),
                    elapsed,
                )

        logger.info("=== Finished ===")
        logger.info(self.credit.summary())
        self._save_summary()

    # ── Single model iteration ────────────────────────────────────────────────

    def _run_one(self, model_id: int) -> None:
        cfg = self.config

        # 1. Generate model
        gen_model: Optional[GeneratedModel] = None
        for _attempt in range(10):
            gen_model = self.search_space.generate()
            if gen_model is not None:
                break
        if gen_model is None:
            logger.debug("Model %d: all generation attempts failed.", model_id)
            return

        self._n_generated += 1

        # 2. Generate inputs (base + mutations)
        all_inputs = self.mutator.generate_all(
            gen_model.input_shape,
            gen_model.input_dtype,
            num_mutations=cfg.num_mutations_per_model,
        )

        # 3. Score with oracle (take max across inputs)
        best_report: Optional[OracleReport] = None
        for feed in all_inputs:
            report = self.oracle.score(gen_model.onnx_model, feed, model_id)
            if best_report is None or report.total_score > best_report.total_score:
                best_report = report

        if best_report is None:
            return

        score = best_report.total_score
        self._total_score += score
        if best_report.crashes:
            self._n_crashes += 1

        # 4. Update credit assignment
        self.credit.update(gen_model.pattern_sequence, score)

        # 5. Update search space policies
        self.search_space.update_policies(
            self.credit.category_utilities(),
            self.credit.pattern_utilities(),
        )

        # 6. Log / save bugs
        if score >= cfg.bug_score_threshold:
            self._n_bugs += 1
            if cfg.verbose:
                logger.info(
                    "BUG #%d  model=%d  score=%.4f  patterns=%s  crashes=%s",
                    self._n_bugs, model_id, score,
                    gen_model.pattern_sequence,
                    best_report.crashes,
                )
            if cfg.save_bugs:
                self._save_bug(gen_model, best_report, model_id)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_bug(
        self,
        gen_model: GeneratedModel,
        report: OracleReport,
        model_id: int,
    ) -> None:
        out_dir = self.config.output_dir
        prefix  = os.path.join(out_dir, f"bug_{model_id:06d}")

        # Save ONNX model
        onnx_path = f"{prefix}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(gen_model.onnx_model.SerializeToString())

        # Save report JSON
        report_path = f"{prefix}_report.json"
        data = {
            "model_id":        model_id,
            "total_score":     report.total_score,
            "s_diff":          report.s_diff,
            "delta_opt":       report.delta_opt,
            "crashes":         report.crashes,
            "errors":          report.errors,
            "pattern_sequence": gen_model.pattern_sequence,
            "input_shape":      gen_model.input_shape,
        }
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_summary(self) -> None:
        out_path = os.path.join(self.config.output_dir, "summary.json")
        data = {
            "num_generated":  self._n_generated,
            "num_bugs":       self._n_bugs,
            "num_crashes":    self._n_crashes,
            "avg_score":      self._total_score / max(self._n_generated, 1),
            "top_patterns":   [
                {"category": c, "name": n, "avg_reward": float(r), "count": cnt}
                for c, n, r, cnt in self.credit.top_patterns(20)
            ],
        }
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Summary saved to %s", out_path)
