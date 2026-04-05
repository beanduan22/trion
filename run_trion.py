#!/usr/bin/env python3
"""
Trion entry point.

Usage:
    python run_trion.py [OPTIONS]

Options:
    --num-models N          Total test models (default: 1000)
    --pattern-budget K      Patterns per model (default: 6)
    --backends B [B ...]    Target backends: tvm onnxruntime tensorrt xla
    --output-dir DIR        Where to save bug reports (default: trion_results)
    --seed S                RNG seed (default: 42)
    --tolerance T           Discrepancy threshold δ (default: 1e-3)
    --no-save               Disable saving ONNX + reports
    --quiet                 Reduce log output
"""
import argparse
import sys

from trion.config import TrionConfig
from trion.runner import TrionRunner


def parse_args():
    p = argparse.ArgumentParser(description="Trion DL Compiler Tester")
    p.add_argument("--num-models",     type=int,   default=1000)
    p.add_argument("--pattern-budget", type=int,   default=6)
    p.add_argument("--backends",       nargs="+",
                   default=["tvm", "onnxruntime", "tensorrt", "xla"],
                   choices=["tvm", "onnxruntime", "tensorrt", "xla"])
    p.add_argument("--output-dir",     type=str,   default="trion_results")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--tolerance",      type=float, default=1e-3)
    p.add_argument("--tvm-target",     type=str,   default="llvm")
    p.add_argument("--tvm-opt-level",  type=int,   default=3)
    p.add_argument("--fp16",           action="store_true")
    p.add_argument("--no-save",        action="store_true")
    p.add_argument("--quiet",          action="store_true")
    p.add_argument("--bug-threshold",  type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()

    config = TrionConfig(
        num_models          = args.num_models,
        pattern_budget      = args.pattern_budget,
        target_backends     = args.backends,
        output_dir          = args.output_dir,
        seed                = args.seed,
        tolerance           = args.tolerance,
        tvm_target          = args.tvm_target,
        tvm_opt_level       = args.tvm_opt_level,
        tensorrt_fp16       = args.fp16,
        save_bugs           = not args.no_save,
        verbose             = not args.quiet,
        bug_score_threshold = args.bug_threshold,
    )

    runner = TrionRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
