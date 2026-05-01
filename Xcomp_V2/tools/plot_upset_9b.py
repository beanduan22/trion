#!/usr/bin/env python3
"""
UpSet plots for the 9-backend smoke probe.

Two figures:
  fail_upset.png — intersections of FAILING pattern sets per backend
                   (the bug-hunting view: which backends co-fail)
  pass_upset.png — intersections of PASSING pattern sets per backend
                   (the coverage view: which backends share clean coverage)

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

BACKENDS = [
    "onnxruntime", "tensorrt", "tvm", "openvino",
    "torchscript", "torch_compile", "xla", "tensorflow", "tflite",
]


def _load_indicator_df(src: Path, status_target: str) -> pd.DataFrame:
    """Return a DataFrame indexed by pattern, one bool column per backend.

    A True value means the pattern matched ``status_target`` on that backend.
    Patterns marked 'incompatible' on a backend are treated as False (not in
    the set), since they could not even be exercised.
    """
    cache = json.loads(src.read_text())
    rows = {}
    for pname, per in cache["patterns"].items():
        row = {}
        for b in BACKENDS:
            entry = per.get(b, {})
            status = entry.get("status") if isinstance(entry, dict) else None
            row[b] = (status == status_target)
        rows[pname] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=BACKENDS)


def _plot(df: pd.DataFrame, title: str, out_path: Path, min_size: int) -> None:
    nonzero_rows = df.any(axis=1).sum()
    if nonzero_rows == 0:
        print(f"  [skip] {out_path.name}: no patterns matched")
        return

    upset_input = from_indicators(BACKENDS, data=df[df.any(axis=1)])
    fig = plt.figure(figsize=(14, 7))
    UpSet(
        upset_input,
        subset_size="count",
        min_subset_size=min_size,
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by="-input",
    ).plot(fig=fig)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(_DEFAULT_SRC))
    ap.add_argument("--out-dir", default=str(_OUT_DIR))
    ap.add_argument(
        "--min-size", type=int, default=1,
        help="Hide intersections whose count is below this threshold.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {src}")
    fail_df = _load_indicator_df(src, "fail")
    pass_df = _load_indicator_df(src, "pass")

    n_patterns = len(fail_df)
    n_any_fail = int(fail_df.any(axis=1).sum())
    n_all_pass = int(pass_df.all(axis=1).sum())
    print(f"[stat] patterns={n_patterns}  any_fail={n_any_fail}  all_pass={n_all_pass}")

    _plot(
        fail_df,
        f"Pattern-failure intersections across 9 backends "
        f"(N={n_patterns}, ≥1 fail = {n_any_fail})",
        out_dir / "fail_upset.png",
        min_size=args.min_size,
    )
    _plot(
        pass_df,
        f"Pattern-pass intersections across 9 backends "
        f"(N={n_patterns}, all-pass = {n_all_pass})",
        out_dir / "pass_upset.png",
        min_size=args.min_size,
    )

    csv_path = out_dir / "pattern_status_indicators.csv"
    combo = fail_df.add_suffix("__fail").join(pass_df.add_suffix("__pass"))
    combo.to_csv(csv_path)
    print(f"  [ok] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
