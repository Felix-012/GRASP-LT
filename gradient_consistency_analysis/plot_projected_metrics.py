#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


COSINE_LIKE_METRICS = {
    "alignment",
    "gc_paper",
    "mean_grad_cos",
    "intra_cos",
    "inter_cos",
}

SPECIAL_RANGES = {
    "gc_paper": (0.0, 1.0),
    "conflict_all": (0.0, 1.0),
    "conflict_sub": (0.0, 1.0),
    "evr_all": (0.0, 1.0),
    "evr_sub": (0.0, 1.0),
    "intra_minus_inter": (-2.0, 2.0),
}

METRIC_DISPLAY_NAMES = {
    "alignment": "Gradient Consistency",
    "gc_paper": "Gradient Consistency",
    "mean_grad_cos": "Mean Gradient Cosine",
    "intra_cos": "Intra-Group Cosine",
    "inter_cos": "Inter-Group Cosine",
    "intra_minus_inter": "Intra- vs Inter-Group Cosine",
    "conflict_all": "Conflict Fraction (All)",
    "conflict_sub": "Conflict Fraction (Subset)",
    "evr_all": "Explained Variance Ratio (All)",
    "evr_sub": "Explained Variance Ratio (Subset)",
}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    w = int(window)
    if w <= 1:
        return values
    w = min(w, len(values))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(values, kernel, mode="same")


def load_curve(csv_path: Path, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    per_step: Dict[int, List[float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if metric not in (reader.fieldnames or []):
            return np.array([]), np.array([])
        for row in reader:
            try:
                step = int(row["step"])
                val = float(row[metric])
            except Exception:
                continue
            if np.isfinite(val):
                per_step.setdefault(step, []).append(val)

    if not per_step:
        return np.array([]), np.array([])

    steps = np.array(sorted(per_step.keys()), dtype=np.int64)
    mean_vals = np.array([float(np.mean(per_step[s])) for s in steps], dtype=np.float64)
    return steps, mean_vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare projected-gradient training metrics across runs."
    )
    parser.add_argument("--csvs", nargs="+", required=True, help="Projected-metrics CSV paths.")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names in the same order as --csvs.",
    )
    parser.add_argument("--metrics", nargs="+", required=True, help="Metric columns to plot.")
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for per-metric plots.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="projected_metric_compare",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Projected Metric Over Training",
        help="Prefix in plot titles (used only when --show-title is set).",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Show a title above each metric plot.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window size (>=1).",
    )
    return parser.parse_args()


def metric_ylim(metric: str) -> Optional[Tuple[float, float]]:
    if metric in SPECIAL_RANGES:
        return SPECIAL_RANGES[metric]
    if metric in COSINE_LIKE_METRICS:
        return (-1.0, 1.0)
    return None


def safe_name(metric: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in metric)


def metric_label(metric: str) -> str:
    if metric in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric]
    return metric.replace("_", " ").title()


def main() -> None:
    args = parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    if args.names is not None and len(args.names) not in (0, len(csv_paths)):
        raise ValueError("If --names is provided, it must have the same length as --csvs.")
    names = args.names if args.names else [p.stem for p in csv_paths]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Light, publication-friendly style.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    for metric in args.metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        plotted = 0
        for name, path in zip(names, csv_paths):
            steps, vals = load_curve(path, metric)
            if len(steps) == 0:
                continue
            vals = moving_average(vals, max(1, int(args.smooth)))
            ax.plot(steps, vals, linewidth=2.3, label=name)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            print(f"[skip] no valid curves for metric={metric!r}")
            continue

        if args.show_title:
            ax.set_title(f"{args.title_prefix}: {metric_label(metric)}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric_label(metric))
        yr = metric_ylim(metric)
        if yr is not None:
            ax.set_ylim(yr[0], yr[1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.35, linestyle="--", linewidth=0.7)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

        out_png = outdir / f"{args.prefix}_{safe_name(metric)}.png"
        fig.savefig(out_png, dpi=250)
        fig.savefig(out_png.with_suffix(".pdf"))
        plt.close(fig)
        print(f"[ok] wrote: {out_png}")


if __name__ == "__main__":
    main()
