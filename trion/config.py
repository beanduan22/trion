from dataclasses import dataclass, field
from typing import List


@dataclass
class TrionConfig:
    # ── Search space ──────────────────────────────────────────────────────────
    pattern_budget: int = 5          # K: patterns per model
    num_models: int = 1500            # total test models
    seed: int = 42

    # ── Feedback (UCB) ────────────────────────────────────────────────────────
    exploration_coefficient: float = 1.0   # λ
    epsilon: float = 1e-8                  # numerical stability ε

    # ── Oracle ────────────────────────────────────────────────────────────────
    tolerance: float = 1e-2                # δ for discrepancy threshold (L2, 10⁻²)

    # ── Backends ──────────────────────────────────────────────────────────────
    reference_backend: str = "pytorch_eager"
    target_backends: List[str] = field(
        default_factory=lambda: [
            "onnxruntime", "torchscript", "torch_compile",
            "xla", "tvm",
            "tensorflow", "openvino",
        ]
    )

    # ── TVM ───────────────────────────────────────────────────────────────────
    tvm_target: str = "llvm"
    tvm_opt_level: int = 3

    # ── TensorRT ──────────────────────────────────────────────────────────────
    tensorrt_fp16: bool = False

    # ── Input ─────────────────────────────────────────────────────────────────
    batch_size: int = 1
    default_channels: int = 3
    default_spatial_size: int = 32   # H = W for initial 4-D tensors
    default_seq_len: int = 16        # sequence length for 3-D tensors

    # ── Input mutations ───────────────────────────────────────────────────────
    num_mutations_per_model: int = 2

    # ── Checkpointing ─────────────────────────────────────────────────────────
    # Every `checkpoint_every` models, snapshot the current bug count, the
    # latest summary.json, and the per-compiler coverage so partial results
    # survive Ctrl-C and we can plot bug-yield over time.
    checkpoint_every: int = 100

    # ── Model size guard ──────────────────────────────────────────────────────
    max_model_bytes: int = 30 * 1024 * 1024   # skip models larger than 30 MB

    # ── Sampling strategy ─────────────────────────────────────────────────────
    # "ucb"     — original UCB-driven policy (concentrates on high-reward patterns)
    # "uniform" — every compatible pattern sampled with equal probability
    # "mixed"   — uniform for the first sweep of patterns, UCB afterward
    sampling_strategy: str = "uniform"

    # ── Pattern-compat cache ──────────────────────────────────────────────────
    # Path to a JSON produced by `tools/check_pattern_compat.py`.  If set,
    # patterns unsupported by any active target backend are excluded.
    pattern_compat_json: str = "campaign_results/pattern_compat_v4_real_tvm.json"

    # ── Resume ────────────────────────────────────────────────────────────────
    # Skip the first N models by fast-forwarding the RNG without scoring.
    # Set to the last completed model_id + 1 to resume a killed campaign.
    start_model: int = 0

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "trion_results"
    save_bugs: bool = True
    save_all: bool = False       # save every model, not just bugs
    verbose: bool = True
    bug_score_threshold: float = 0.1   # score above which we log a "bug"
