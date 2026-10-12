#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    w = int(window)
    if w <= 1:
        return values
    # Keep smoothed curve length equal to input length.
    w = min(w, len(values))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(values, kernel, mode="same")


def load_curve(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    per_step: Dict[int, List[float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row["step"])
                val = float(row["alignment"])
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
        description="Compare gradient consistency curves across training runs."
    )
    parser.add_argument("--csvs", nargs="+", required=True, help="CSV paths from gc_train_mode runs.")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names in the same order as --csvs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="projected_alignment_compare.png",
        help="Output plot path (.png recommended; a .pdf is also written next to it).",
    )
    parser.add_argument("--title", type=str, default="Gradient Consistency Over Training")
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window size (>=1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    if args.names is not None and len(args.names) not in (0, len(csv_paths)):
        raise ValueError("If --names is provided, it must have the same length as --csvs.")
    names = args.names if args.names else [p.stem for p in csv_paths]

    plt.figure(figsize=(9, 4.5))
    plotted = 0
    for name, path in zip(names, csv_paths):
        steps, vals = load_curve(path)
        if len(steps) == 0:
            continue
        vals = moving_average(vals, max(1, int(args.smooth)))
        plt.plot(steps, vals, linewidth=2.0, label=name)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No valid curves found in the provided CSV files.")

    plt.title(args.title)
    plt.xlabel("Training step")
    plt.ylabel("Gradient Consistency")
    plt.ylim(-1.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    if out_path.suffix.lower() == ".png":
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


if __name__ == "__main__":
    main()
