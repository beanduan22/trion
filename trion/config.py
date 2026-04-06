from dataclasses import dataclass, field
from typing import List


@dataclass
class TrionConfig:
    # ── Search space ──────────────────────────────────────────────────────────
    pattern_budget: int = 6          # K: patterns per model
    num_models: int = 1000            # total test models
    seed: int = 42

    # ── Feedback (UCB) ────────────────────────────────────────────────────────
    exploration_coefficient: float = 1.0   # λ
    epsilon: float = 1e-8                  # numerical stability ε

    # ── Oracle ────────────────────────────────────────────────────────────────
    tolerance: float = 0.01               # δ for discrepancy threshold (1% relative diff)

    # ── Backends ──────────────────────────────────────────────────────────────
    reference_backend: str = "pytorch_eager"
    target_backends: List[str] = field(
        default_factory=lambda: [
            "onnxruntime", "torchscript", "torch_compile",
            "xla", "tvm",
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
    num_mutations_per_model: int = 3

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "trion_results"
    save_bugs: bool = True
    save_all: bool = False       # save every model, not just bugs
    verbose: bool = True
    bug_score_threshold: float = 0.1   # score above which we log a "bug"
