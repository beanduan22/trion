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
        self._n_generated   = 0
        self._n_bugs        = 0     # models with score >= threshold
        self._n_crashes     = 0     # models with at least one crash
        self._n_compiler_crashes = 0  # backend-type crashes only
        self._total_score   = 0.0

        # Bug-type counters: error_signature → count
        import collections
        self._crash_type_counts: dict = collections.defaultdict(int)  # sig → count
        self._crash_backend_counts: dict = collections.defaultdict(int)  # backend → count

        if cfg.save_bugs:
            os.makedirs(cfg.output_dir, exist_ok=True)
            # Separate subdir for crash-only compiler bugs
            self._crash_dir = os.path.join(cfg.output_dir, "compiler_crashes")
            os.makedirs(self._crash_dir, exist_ok=True)

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

        # Track crash type counts (every occurrence, not just first)
        for label, ctype in best_report.crash_info.items():
            bname = label.split('+')[0].split('-')[0].replace('ref:', '')
            sig = best_report.error_signatures.get(label, "unknown")
            if ctype == "backend":
                self._n_compiler_crashes += 1
                self._crash_type_counts[f"[{bname}] {sig}"] += 1
                self._crash_backend_counts[bname] += 1

        # 4. Update credit assignment
        # Determine dominant crash type for this model
        crash_types = list(best_report.crash_info.values())
        all_frontend = crash_types and all(t == "frontend" for t in crash_types)
        has_backend_crash = any(t == "backend" for t in crash_types)

        if score > 0:
            # Genuine numerical discrepancy — reward patterns
            self.credit.update(gen_model.pattern_sequence, score)
        elif has_backend_crash:
            # Real backend crash, no discrepancy signal — small penalty
            self.credit.update_crash(gen_model.pattern_sequence, "backend")
        elif all_frontend:
            # Pure frontend/tracing failure — don't penalize patterns at all
            self.credit.update_crash(gen_model.pattern_sequence, "frontend")
        else:
            # Clean run with zero discrepancy — still update with 0 reward
            self.credit.update(gen_model.pattern_sequence, 0.0)

        # 5. Update search space policies
        self.search_space.update_policies(
            self.credit.category_utilities(),
            self.credit.pattern_utilities(),
        )

        # 6. Log / save bugs
        if cfg.save_all and cfg.save_bugs:
            # Save every model regardless of score
            self._save_bug(gen_model, best_report, model_id)
            if score >= cfg.bug_score_threshold:
                self._n_bugs += 1

        elif score >= cfg.bug_score_threshold:
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

        # Always save models with backend (compiler) crashes, even if score=0
        # These are genuine compiler bugs — log every occurrence
        elif cfg.save_bugs and has_backend_crash:
            if cfg.verbose:
                logger.info(
                    "COMPILER CRASH  model=%d  backends=%s",
                    model_id,
                    [l for l, t in best_report.crash_info.items() if t == "backend"],
                )
            self._save_compiler_crash(gen_model, best_report, model_id)

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
        crash_types = list(report.crash_info.values())
        data = {
            "model_id":          model_id,
            "total_score":       report.total_score,
            "n_valid_comparisons": report.n_valid_comparisons,
            "s_diff":            report.s_diff,
            "delta_opt":         report.delta_opt,
            "crashes":           report.crashes,
            "crash_info":        report.crash_info,
            "crash_counts": {
                "frontend": crash_types.count("frontend"),
                "backend":  crash_types.count("backend"),
            },
            "errors":            report.errors,
            "pattern_sequence":  gen_model.pattern_sequence,
            "input_shape":       gen_model.input_shape,
        }
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_compiler_crash(
        self,
        gen_model: GeneratedModel,
        report: OracleReport,
        model_id: int,
    ) -> None:
        """Save a model that caused a genuine compiler (backend) crash."""
        prefix = os.path.join(self._crash_dir, f"crash_{model_id:06d}")
        with open(f"{prefix}.onnx", "wb") as f:
            f.write(gen_model.onnx_model.SerializeToString())
        crash_types = list(report.crash_info.values())
        data = {
            "model_id":         model_id,
            "total_score":      report.total_score,
            "crashes":          report.crashes,
            "crash_info":       report.crash_info,
            "error_signatures": report.error_signatures,
            "crash_counts": {
                "frontend": crash_types.count("frontend"),
                "backend":  crash_types.count("backend"),
            },
            "errors":           report.errors,
            "pattern_sequence": gen_model.pattern_sequence,
            "input_shape":      gen_model.input_shape,
        }
        with open(f"{prefix}_report.json", "w") as f:
            json.dump(data, f, indent=2)

    def _save_summary(self) -> None:
        out_path = os.path.join(self.config.output_dir, "summary.json")

        # Full pattern usage — every registered pattern
        all_usage = [
            {"category": c, "name": n, "avg_reward": float(r), "count": cnt}
            for c, n, r, cnt in self.credit.all_pattern_usage()
        ]
        # Patterns never sampled
        never_used = [p for p in all_usage if p["count"] == 0]
        used = [p for p in all_usage if p["count"] > 0]

        data = {
            "num_generated":          self._n_generated,
            "num_divergence_bugs":    self._n_bugs,
            "num_compiler_crashes":   self._n_compiler_crashes,
            "num_any_crash_models":   self._n_crashes,
            "avg_score":              self._total_score / max(self._n_generated, 1),
            # Pattern coverage
            "patterns_total":         len(all_usage),
            "patterns_used":          len(used),
            "patterns_never_used":    len(never_used),
            "never_used_patterns":    [f"{p['category']}/{p['name']}" for p in never_used],
            # Per-pattern full table (sorted by count desc)
            "pattern_usage": sorted(all_usage, key=lambda p: -p["count"]),
            # Compiler crash breakdown by backend and error type
            "compiler_crash_by_backend": dict(self._crash_backend_counts),
            "compiler_crash_types": dict(
                sorted(self._crash_type_counts.items(), key=lambda x: -x[1])
            ),
        }
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Summary saved to %s", out_path)
