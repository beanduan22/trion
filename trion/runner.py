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
import collections
import json
import logging
import os
import time
from typing import Dict, Optional, Set

import numpy as np
import onnx

from .config import TrionConfig
from .coverage_tracker import CoverageTracker, format_report as _fmt_cov
from .feedback.credit_assignment import CreditAssignment
from .generation.search_space import PatternAwareSearchSpace, GeneratedModel
from .mutation.input_mutator import InputMutator
from .oracle.oracle import DiscrepancyOracle, OracleReport
from .patterns.library import OTPLibrary
from .reproducer import make_reproducer

logger = logging.getLogger(__name__)


def _extract_ops(model: onnx.ModelProto) -> Set[str]:
    """Return the set of ONNX op_type strings used in a model's graph."""
    return {node.op_type for node in model.graph.node}


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

        os.makedirs(cfg.output_dir, exist_ok=True)

        _fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _root = logging.getLogger()
        _root.setLevel(logging.INFO if cfg.verbose else logging.WARNING)

        # Console handler (already present if basicConfig was called before;
        # add one only when the root logger has no handlers yet)
        if not _root.handlers:
            _ch = logging.StreamHandler()
            _ch.setFormatter(_fmt)
            _root.addHandler(_ch)

        # File handler — always write to trion.log inside output_dir
        _log_path = os.path.join(cfg.output_dir, "trion.log")
        _fh = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
        _fh.setFormatter(_fmt)
        _fh.setLevel(logging.DEBUG)   # capture everything to file
        _root.addHandler(_fh)
        logger.info("Logging to %s", _log_path)

        rng = np.random.default_rng(cfg.seed)

        self.library     = OTPLibrary(
            compat_json=cfg.pattern_compat_json,
            active_backends=list(cfg.target_backends),
        )
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
        self._crash_type_counts: dict = collections.defaultdict(int)  # sig → count
        self._crash_backend_counts: dict = collections.defaultdict(int)  # backend → count

        # ONNX op coverage: tracks which ONNX operators are exercised
        # across all generated models and specifically in bug-triggering models.
        self._op_counts: Dict[str, int] = collections.defaultdict(int)      # op → #models seen in
        self._op_bug_counts: Dict[str, int] = collections.defaultdict(int)  # op → #bug models
        self._op_crash_counts: Dict[str, int] = collections.defaultdict(int) # op → #crash models

        if cfg.save_bugs:
            os.makedirs(cfg.output_dir, exist_ok=True)
            # Separate subdir for crash-only compiler bugs
            self._crash_dir = os.path.join(cfg.output_dir, "compiler_crashes")
            os.makedirs(self._crash_dir, exist_ok=True)

        # Per-compiler coverage tracker (line + branch).  We measure ONLY the
        # backends under test, never the whole environment.
        self.coverage = CoverageTracker(list(cfg.target_backends))

    # ── Public entry-point ────────────────────────────────────────────────────

    def run(self) -> None:
        import datetime
        cfg = self.config
        t0 = time.time()
        start_dt = datetime.datetime.now()

        logger.info("=== Trion  started %s ===", start_dt.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info(self.library.summary())
        logger.info(self.oracle.summary())

        # Fast-forward RNG to reproduce the exact state at start_model by
        # replaying generation (no scoring) for models 0..start_model-1.
        if cfg.start_model > 0:
            logger.info(
                "Resuming from model %d — fast-forwarding RNG through %d models …",
                cfg.start_model, cfg.start_model,
            )
            for i in range(cfg.start_model):
                self._rng_advance(i)
            logger.info("RNG fast-forward complete. Resuming scoring from model %d.", cfg.start_model)

        logger.info("Starting %d model generations (models %d–%d) …",
                    cfg.num_models - cfg.start_model, cfg.start_model, cfg.num_models - 1)

        # Begin per-compiler coverage tracking BEFORE the first backend call.
        self.coverage.start()

        try:
            for i in range(cfg.start_model, cfg.num_models):
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

                # Periodic checkpoint: snapshot summary + bug/coverage state.
                if (
                    cfg.checkpoint_every > 0
                    and (i + 1) % cfg.checkpoint_every == 0
                    and (i + 1) < cfg.num_models
                    and i >= cfg.start_model
                ):
                    self._write_checkpoint(i + 1, time.time() - t0)
        finally:
            # Always stop coverage so the report is produced even on Ctrl-C.
            self.coverage.stop()

        elapsed_total = time.time() - t0
        end_dt = datetime.datetime.now()
        logger.info(
            "=== Finished  %s  (elapsed %.1fs / %.1f min) ===",
            end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_total, elapsed_total / 60,
        )
        logger.info(self.credit.summary())
        self._save_summary(start_dt=start_dt, elapsed_s=elapsed_total)
        self._save_op_coverage()
        self._report_coverage()

    # ── RNG fast-forward (resume support) ────────────────────────────────────

    def _rng_advance(self, model_id: int) -> None:
        """Replay generation RNG calls for model_id without running any backend.

        This advances the shared RNG to the exact same state it would be in
        after a normal _run_one(model_id) call, so that resuming at
        start_model produces identical subsequent models to an uninterrupted run.
        """
        cfg = self.config
        for _attempt in range(10):
            gen_model = self.search_space.generate()
            if gen_model is not None:
                break
        else:
            return
        # Advance input-mutator RNG to match what generate_all() would consume.
        self.mutator.generate_all(
            gen_model.input_shape,
            gen_model.input_dtype,
            num_mutations=cfg.num_mutations_per_model,
        )

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

        # Size guard — skip very large models to prevent OOM
        model_bytes = len(gen_model.onnx_model.SerializeToString())
        if model_bytes > cfg.max_model_bytes:
            logger.debug(
                "Model %d: too large (%d MB), skipping.",
                model_id, model_bytes // (1024 * 1024),
            )
            return

        # ONNX validity check — reject structurally malformed models before
        # running any backend.  Malformed models can produce undefined outputs
        # on pytorch_eager that look like bugs but are just model errors.
        try:
            onnx.checker.check_model(gen_model.onnx_model)
        except onnx.checker.ValidationError as e:
            logger.debug("Model %d: ONNX validation failed (%s) — skipping.", model_id, e)
            return

        self._n_generated += 1

        # Record ONNX op coverage for every valid model
        model_ops = _extract_ops(gen_model.onnx_model)
        for op in model_ops:
            self._op_counts[op] += 1

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
            for op in model_ops:
                self._op_bug_counts[op] += 1
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
            for op in model_ops:
                self._op_crash_counts[op] += 1
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
        onnx_serialized = gen_model.onnx_model.SerializeToString()
        with open(onnx_path, "wb") as f:
            f.write(onnx_serialized)

        # Self-contained Python reproducer (no external file dependency).
        try:
            inputs_np = {
                name: np.asarray(vals, dtype=np.float32).reshape(gen_model.input_shape)
                for name, vals in report.bug_inputs.items()
            }
            src = make_reproducer(
                model_id          = model_id,
                onnx_bytes        = onnx_serialized,
                inputs            = inputs_np,
                pattern_sequence  = gen_model.pattern_sequence,
                tolerance         = self.config.tolerance,
                s_diff            = dict(report.s_diff),
                delta_opt         = dict(report.delta_opt),
                expected_outputs  = dict(report.expected_outputs),
                buggy_outputs     = dict(report.buggy_outputs),
            )
            with open(f"{prefix}_repro.py", "w") as f:
                f.write(src)
        except Exception as exc:
            logger.warning(
                "Reproducer generation failed for bug %d: %s",
                model_id, exc,
            )

        # Save report JSON
        report_path = f"{prefix}_report.json"
        crash_types = list(report.crash_info.values())
        # Pairwise divergence uses tuple keys → stringify for JSON.
        pairwise_str = {
            f"{k[0]}|{k[1]}": v
            for k, v in getattr(report, "pairwise_divergence", {}).items()
        }
        wp = getattr(report, "worst_pair", None)
        data = {
            "model_id":          model_id,
            "total_score":       report.total_score,
            # New oracle fields (empty on old OracleReport instances).
            "worst_pair":        list(wp) if wp is not None else None,
            "pairwise_divergence": pairwise_str,
            "suspect_backend":   getattr(report, "suspect_backend", None),
            "suspect_distance_to_eager":
                                 getattr(report, "suspect_distance_to_eager", 0.0),
            "eager_helper_used": getattr(report, "eager_helper_used", False),
            "n_valid_targets":   getattr(report, "n_valid_targets", 0),
            "shared_infra_warning":
                                 getattr(report, "shared_infra_warning", None),
            "skipped_targets":   getattr(report, "skipped_targets", {}),
            # Frontend-validation gate (Oracle 1 connection-correctness layer)
            "frontend_gate_ref":
                                 getattr(report, "frontend_gate_ref", None),
            "frontend_gate_passed":
                                 getattr(report, "frontend_gate_passed", {}),
            "frontend_gate_rejected":
                                 getattr(report, "frontend_gate_rejected", {}),
            # Legacy fields kept for back-compat.
            "n_valid_comparisons": report.n_valid_comparisons,
            "s_diff":            report.s_diff,
            "delta_opt":         report.delta_opt,
            "noopt_valid":       report.noopt_valid,
            "crashes":           report.crashes,
            "crash_info":        report.crash_info,
            "crash_counts": {
                "frontend": crash_types.count("frontend"),
                "backend":  crash_types.count("backend"),
            },
            "errors":            report.errors,
            "pattern_sequence":  gen_model.pattern_sequence,
            "input_shape":       gen_model.input_shape,
            # Bug evidence — exact arrays for reproduction
            "bug_inputs":        report.bug_inputs,
            "expected_outputs":  report.expected_outputs,
            "buggy_outputs":     report.buggy_outputs,
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

    def _save_summary(self, start_dt=None, elapsed_s: float = 0.0) -> None:
        import datetime
        out_path = os.path.join(self.config.output_dir, "summary.json")

        # Full pattern usage — every registered pattern
        all_usage = [
            {"category": c, "name": n, "avg_reward": float(r), "count": cnt}
            for c, n, r, cnt in self.credit.all_pattern_usage()
        ]
        # Patterns never sampled
        never_used = [p for p in all_usage if p["count"] == 0]
        used = [p for p in all_usage if p["count"] > 0]

        now = datetime.datetime.now()
        data = {
            "start_time":             start_dt.strftime("%Y-%m-%d %H:%M:%S") if start_dt else None,
            "end_time":               now.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds":        round(elapsed_s, 1),
            "elapsed_minutes":        round(elapsed_s / 60, 2),
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

    def _write_checkpoint(self, models_done: int, elapsed_s: float) -> None:
        """Snapshot bugs/coverage/summary at every checkpoint_every interval.

        Writes:
            <output_dir>/checkpoints/cp_NNNNNN/
                summary.json     - partial campaign-wide stats
                bug_index.json   - list of bug ids saved so far
                coverage.json    - per-backend Python coverage at this point
        """
        cfg = self.config
        cp_dir = os.path.join(cfg.output_dir, "checkpoints",
                              f"cp_{models_done:06d}")
        os.makedirs(cp_dir, exist_ok=True)

        try:
            partial_summary = {
                "models_done": models_done,
                "elapsed_seconds": round(elapsed_s, 1),
                "n_generated": self._n_generated,
                "n_divergence_bugs": self._n_bugs,
                "n_compiler_crashes": self._n_compiler_crashes,
                "n_any_crash_models": self._n_crashes,
                "avg_score": (
                    self._total_score / max(self._n_generated, 1)
                ),
                "compiler_crash_by_backend": dict(self._crash_backend_counts),
            }
            with open(os.path.join(cp_dir, "summary.json"), "w") as f:
                json.dump(partial_summary, f, indent=2)

            # Bug ids saved so far (for cross-checkpoint diffs)
            bug_ids = sorted(
                f for f in os.listdir(cfg.output_dir)
                if f.startswith("bug_") and f.endswith(".onnx")
            )
            with open(os.path.join(cp_dir, "bug_index.json"), "w") as f:
                json.dump(bug_ids, f, indent=2)

            # Per-backend Python coverage snapshot. C++ coverage is deferred
            # to the post-run aggregator and only meaningful at the very end.
            try:
                cov = self.coverage.report()
                with open(os.path.join(cp_dir, "coverage.json"), "w") as f:
                    json.dump(cov, f, indent=2)
            except Exception as exc:
                logger.debug("checkpoint coverage capture failed: %s", exc)

            # ONNX op coverage snapshot at this checkpoint
            try:
                op_snap = {
                    "unique_ops_seen": len(self._op_counts),
                    "op_counts": dict(
                        sorted(self._op_counts.items(), key=lambda x: -x[1])
                    ),
                    "op_bug_counts": dict(self._op_bug_counts),
                }
                with open(os.path.join(cp_dir, "op_coverage.json"), "w") as f:
                    json.dump(op_snap, f, indent=2)
            except Exception as exc:
                logger.debug("checkpoint op coverage capture failed: %s", exc)

            logger.info(
                "[checkpoint %d] bugs=%d crashes=%d elapsed=%.0fs → %s",
                models_done, self._n_bugs, self._n_compiler_crashes,
                elapsed_s, cp_dir,
            )
        except Exception as exc:
            logger.warning("Checkpoint write failed: %s", exc)

    def _save_op_coverage(self) -> None:
        """Save ONNX op-level coverage: which ops were exercised, which triggered bugs."""
        # All ONNX ops referenced by the standard (opset 18 core set).
        # Ops not seen in any generated model are listed as uncovered.
        _ONNX_CORE_OPS = {
            "Abs", "Acos", "Acosh", "Add", "And", "ArgMax", "ArgMin", "Asin",
            "Asinh", "Atan", "Atanh", "AveragePool", "BatchNormalization",
            "BitShift", "Cast", "CastLike", "Ceil", "Celu", "Clip",
            "Compress", "Concat", "ConcatFromSequence", "Constant",
            "ConstantOfShape", "Conv", "ConvInteger", "ConvTranspose",
            "Cos", "Cosh", "CumSum", "DepthToSpace", "DequantizeLinear",
            "Det", "Div", "Dropout", "DynamicQuantizeLinear", "Einsum",
            "Elu", "Equal", "Erf", "Exp", "Expand", "EyeLike", "Flatten",
            "Floor", "GRU", "Gather", "GatherElements", "GatherND", "Gemm",
            "GlobalAveragePool", "GlobalLpPool", "GlobalMaxPool", "Greater",
            "GreaterOrEqual", "GroupNormalization", "HardSigmoid", "HardSwish",
            "Hardmax", "Identity", "If", "InstanceNormalization", "IsInf",
            "IsNaN", "LRN", "LSTM", "LayerNormalization", "LeakyRelu", "Less",
            "LessOrEqual", "Log", "LogSoftmax", "Loop", "LpNormalization",
            "LpPool", "MatMul", "MatMulInteger", "Max", "MaxPool", "MaxUnpool",
            "Mean", "MeanVarianceNormalization", "Min", "Mish", "Mod", "Mul",
            "Multinomial", "Neg", "NegativeLogLikelihoodLoss", "NonMaxSuppression",
            "NonZero", "Not", "OneHot", "Optional", "OptionalGetElement",
            "OptionalHasElement", "Or", "PRelu", "Pad", "Pow",
            "QLinearConv", "QLinearMatMul", "QuantizeLinear", "RNN",
            "RandomNormal", "RandomNormalLike", "RandomUniform",
            "RandomUniformLike", "Range", "Reciprocal", "ReduceL1", "ReduceL2",
            "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
            "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
            "Relu", "Reshape", "Resize", "ReverseSequence", "RoiAlign",
            "Round", "STFT", "Scan", "ScatterElements", "ScatterND",
            "Selu", "SequenceAt", "SequenceConstruct", "SequenceEmpty",
            "SequenceErase", "SequenceInsert", "SequenceLength", "SequenceMap",
            "Shape", "Shrink", "Sigmoid", "Sign", "Sin", "Sinh", "Size",
            "Slice", "Softmax", "Softplus", "Softsign", "SpaceToDepth",
            "Split", "SplitToSequence", "Sqrt", "Squeeze", "StringNormalizer",
            "Sub", "Sum", "Tan", "Tanh", "TfIdfVectorizer", "ThresholdedRelu",
            "Tile", "TopK", "Transpose", "Trilu", "Unique", "Unsqueeze",
            "Where", "Xor",
        }

        seen_ops = set(self._op_counts.keys())
        covered = seen_ops & _ONNX_CORE_OPS
        uncovered = _ONNX_CORE_OPS - seen_ops
        non_standard = seen_ops - _ONNX_CORE_OPS  # custom/experimental ops

        # Per-op table sorted by frequency
        per_op = {
            op: {
                "models_seen":    self._op_counts[op],
                "bug_models":     self._op_bug_counts.get(op, 0),
                "crash_models":   self._op_crash_counts.get(op, 0),
                "in_onnx_core":   op in _ONNX_CORE_OPS,
            }
            for op in sorted(self._op_counts, key=lambda o: -self._op_counts[o])
        }

        data = {
            "onnx_core_ops_total":     len(_ONNX_CORE_OPS),
            "onnx_core_ops_covered":   len(covered),
            "onnx_core_ops_uncovered": len(uncovered),
            "coverage_pct":            round(100.0 * len(covered) / len(_ONNX_CORE_OPS), 1),
            "non_standard_ops_seen":   sorted(non_standard),
            "uncovered_ops":           sorted(uncovered),
            "per_op":                  per_op,
        }

        path = os.path.join(self.config.output_dir, "op_coverage.json")
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "ONNX op coverage: %d/%d core ops exercised (%.1f%%)  →  %s",
            len(covered), len(_ONNX_CORE_OPS),
            100.0 * len(covered) / len(_ONNX_CORE_OPS),
            path,
        )

    def _report_coverage(self) -> None:
        """Print per-compiler line+branch coverage and persist as JSON."""
        try:
            report = self.coverage.report()
        except Exception as exc:
            logger.warning("Coverage report generation failed: %s", exc)
            return
        if not report:
            logger.info("Coverage tracking unavailable.")
            return
        for line in _fmt_cov(report).splitlines():
            logger.info(line)
        try:
            cov_path = os.path.join(self.config.output_dir, "coverage_report.json")
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.coverage.save_json(cov_path)
            logger.info("Coverage report saved to %s", cov_path)
        except Exception as exc:
            logger.warning("Coverage JSON dump failed: %s", exc)
