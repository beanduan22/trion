#!/usr/bin/env python3
"""
UpSet plots for the cross-compiler smoke probe.

Default render: pass-intersection figure across the 8 paper backends, no
title, display labels matching the paper's compiler list. Use ``--mode``
to switch between pass-only / fail-only / both.

Source:
  campaign_v10_results/full_probe_9.progress.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_indicators

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _REPO_ROOT / "campaign_v10_results" / "full_probe_9.progress.json"
_OUT_DIR = _REPO_ROOT / "campaign_v10_results" / "figures"

# Internal backend keys (must match keys in the probe JSON) paired with the
# display label used on the figure. Order = top-to-bottom on the left bar.
BACKEND_LABELS: list[tuple[str, str]] = [
    ("onnxruntime",   "ONNXRuntime"),
    ("tensorrt",      "TensorRT"),
    ("tvm",           "TVM"),
    ("torch_compile", "torch.compile"),
    ("torchscript",   "TorchScript"),
    ("xla",           "Jax-XLA"),
    ("tensorflow",    "TensorFlow-XLA"),
    ("tflite",        "TensorFlow-Lite"),
    ("openvino",      "OpenVINO"),
]
BACKEND_KEYS = [k for k, _ in BACKEND_LABELS]
LABEL_BY_KEY = dict(BACKEND_LABELS)
DISPLAY_LABELS = [v for _, v in BACKEND_LABELS]


def _load_indicator_df(src: Path, status_target: str) -> pd.DataFrame:
    """Return a DataFrame indexed by pattern with one bool column per backend.

    A True value means the pattern's status on that backend equals
    ``status_target``. Patterns marked 'incompatible' or 'unavailable' are
    treated as False (not in the set).
    """
    cache = json.loads(src.read_text())
    rows = {}
    for pname, per in cache["patterns"].items():
        row = {}
        for k in BACKEND_KEYS:
            entry = per.get(k, {})
            status = entry.get("status") if isinstance(entry, dict) else None
            row[LABEL_BY_KEY[k]] = (status == status_target)
        rows[pname] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=DISPLAY_LABELS)


def _plot(df: pd.DataFrame, out_path: Path, min_size: int, title: str | None) -> None:
    nonzero_rows = df.any(axis=1).sum()
    if nonzero_rows == 0:
        print(f"  [skip] {out_path.name}: no patterns matched")
        return

    upset_input = from_indicators(DISPLAY_LABELS, data=df[df.any(axis=1)])
    fig = plt.figure(figsize=(14, 7))
    UpSet(
        upset_input,
        subset_size="count",
        min_subset_size=min_size,
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by="-input",
    ).plot(fig=fig)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(_DEFAULT_SRC))
    ap.add_argument("--out-dir", default=str(_OUT_DIR))
    ap.add_argument(
        "--mode", choices=("pass", "fail", "both"), default="pass",
        help="Which intersection figure to render (default: pass).",
    )
    ap.add_argument(
        "--min-size", type=int, default=1,
        help="Hide intersections whose count is below this threshold.",
    )
    ap.add_argument(
        "--title", default=None,
        help="Optional figure title; default is no title.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {src}")
    print(f"[backends] {' | '.join(DISPLAY_LABELS)}")

    if args.mode in ("pass", "both"):
        pass_df = _load_indicator_df(src, "pass")
        n_all_pass = int(pass_df.all(axis=1).sum())
        print(f"[stat] all-pass patterns = {n_all_pass} / {len(pass_df)}")
        _plot(pass_df, out_dir / "upset_pass_9compilers.png", args.min_size, args.title)

    if args.mode in ("fail", "both"):
        fail_df = _load_indicator_df(src, "fail")
        n_any_fail = int(fail_df.any(axis=1).sum())
        print(f"[stat] any-fail patterns = {n_any_fail} / {len(fail_df)}")
        _plot(fail_df, out_dir / "upset_fail_9compilers.png", args.min_size, args.title)

    if args.mode == "both":
        combo = (
            _load_indicator_df(src, "fail").add_suffix("__fail")
            .join(_load_indicator_df(src, "pass").add_suffix("__pass"))
        )
        csv_path = out_dir / "pattern_status_indicators.csv"
        combo.to_csv(csv_path)
        print(f"  [ok] wrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
