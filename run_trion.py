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
import os
import subprocess
import sys

from trion.config import TrionConfig
from trion.runner import TrionRunner


def parse_args():
    p = argparse.ArgumentParser(description="Trion DL Compiler Tester")
    p.add_argument("--num-models",     type=int,   default=1500)
    p.add_argument("--pattern-budget", type=int,   default=5)
    p.add_argument("--backends",       nargs="+",
                   default=["onnxruntime", "torchscript", "torch_compile",
                            "xla", "tvm", "tensorflow", "openvino"],
                   choices=["tvm", "onnxruntime", "tensorrt", "xla",
                            "torchscript", "torch_compile",
                            "tensorflow", "tflite", "openvino"])
    p.add_argument("--output-dir",     type=str,   default="trion_results")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--tolerance",      type=float, default=1e-2)
    p.add_argument("--tvm-target",     type=str,   default="llvm")
    p.add_argument("--tvm-opt-level",  type=int,   default=3)
    p.add_argument("--fp16",           action="store_true")
    p.add_argument("--no-save",        action="store_true")
    p.add_argument("--save-all",       action="store_true",
                   help="Save every generated model, not just bugs")
    p.add_argument("--quiet",          action="store_true")
    p.add_argument("--bug-threshold",  type=float, default=0.05)
    p.add_argument("--sampling",       type=str, default="uniform",
                   choices=["uniform", "ucb"],
                   help="Pattern sampling strategy (default: uniform for "
                        "broad per-campaign coverage)")
    p.add_argument("--pattern-compat", type=str,
                   default="campaign_results/pattern_compat_v4_real_tvm.json",
                   help="Pattern-compatibility cache JSON produced by "
                        "tools/check_pattern_compat.py (v4 = real Apache TVM)")
    p.add_argument("--start-model", type=int, default=0,
                   help="Skip (fast-forward RNG through) the first N models. "
                        "Use to resume a killed campaign at the correct RNG state.")
    return p.parse_args()


def main():
    args = parse_args()

    # Two-phase execution for C++ coverage aggregation:
    # libgcov only flushes .gcda files at process exit, and its dump
    # symbols are not dlsym-reachable, so we cannot aggregate C++ coverage
    # in-process.  Instead we re-exec ourselves as an inner child; after
    # the child exits (and gcov flushes), the outer parent runs the
    # post-run aggregator which walks the freshly-written .gcda files.
    if os.environ.get("TRION_INNER") != "1":
        env = dict(os.environ)
        env["TRION_INNER"] = "1"
        rc = subprocess.call([sys.executable, __file__] + sys.argv[1:], env=env)
        # Inner has exited → .gcda files are now on disk
        try:
            from trion.coverage_tracker import aggregate_cpp_postrun
            aggregate_cpp_postrun(args.backends, args.output_dir)
        except Exception as exc:
            print(f"[warn] post-run C++ coverage aggregation failed: {exc}",
                  file=sys.stderr)
        sys.exit(rc)

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
        save_all            = args.save_all,
        verbose             = not args.quiet,
        bug_score_threshold = args.bug_threshold,
        sampling_strategy   = args.sampling,
        pattern_compat_json = args.pattern_compat,
        start_model         = args.start_model,
    )

    runner = TrionRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
