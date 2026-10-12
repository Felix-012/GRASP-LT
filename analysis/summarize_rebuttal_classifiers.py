#!/usr/bin/env python3
"""Aggregate the partition-sensitivity classifiers using the paper protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = ("random0", "random1", "random2", "k3", "k5", "oneperlabel", "hmt", "hybrid")
DISPLAY = {
    "paper_grasp_lowdim": "Informed (paper GRASP-LowDim)",
    "random_mean": "Random ×3",
    "random0": "Random 0",
    "random1": "Random 1",
    "random2": "Random 2",
    "k3": "K=3",
    "k5": "K=5",
    "oneperlabel": "One expert per label",
    "hmt": "Literal H/M/T",
    "hybrid": "Static-initialized learned gate",
}
PAPER_MODEL = "grasp_lowdim"


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, sd


def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("No classifier runs supplied")
    labels = list(runs[0]["per_class"])
    if len(labels) != 13:
        raise ValueError(f"Expected the paper's 13 NIH-shared classes, got {len(labels)}: {labels}")
    for run in runs:
        if list(run["per_class"]) != labels:
            raise ValueError("Classifier runs have inconsistent class order")
    macro = [float(run["aggregates"]["macro_f1"]) for run in runs]
    per_class_mean = {
        label: sum(float(run["per_class"][label]["f1"]) for run in runs) / len(runs)
        for label in labels
    }
    return {
        "runs": len(runs),
        "macro": mean_sd(macro),
        "nonzero": sum(value > 0.0 for value in per_class_mean.values()),
        "per_class_mean": per_class_mean,
    }


def load_variant(root: Path, variant: str) -> list[dict]:
    paths = sorted((root / variant).glob("*/seed_*/best_metrics.json"))
    if len(paths) != 3:
        raise ValueError(f"{variant}: expected 3 best_metrics.json files, got {len(paths)}")
    return [json.loads(path.read_text()) for path in paths]


def load_paper(csv_path: Path) -> dict:
    rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == PAPER_MODEL:
                rows.append(row)
    if len(rows) != 9:
        raise ValueError(f"Expected 9 paper GRASP-LowDim rows, got {len(rows)}")
    label_columns = [column for column in rows[0] if column.endswith("_f1") and column != "macro_f1"]
    if len(label_columns) != 13:
        raise ValueError(f"Expected 13 paper label columns, got {label_columns}")
    converted = []
    for row in rows:
        converted.append(
            {
                "aggregates": {"macro_f1": float(row["macro_f1"])},
                "per_class": {
                    column[:-3]: {"f1": float(row[column])} for column in label_columns
                },
            }
        )
    return summarize_runs(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="classifier_results directory")
    parser.add_argument(
        "--paper-csv",
        type=Path,
        default=Path("consolidated_classifier_results.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {variant: load_variant(args.root, variant) for variant in VARIANTS}
    summary = {variant: summarize_runs(payloads) for variant, payloads in runs.items()}
    summary["random_mean"] = summarize_runs(
        [payload for variant in ("random0", "random1", "random2") for payload in runs[variant]]
    )
    summary["paper_grasp_lowdim"] = load_paper(args.paper_csv)

    order = (
        "paper_grasp_lowdim",
        "random_mean",
        "random0",
        "random1",
        "random2",
        "k3",
        "k5",
        "oneperlabel",
        "hmt",
        "hybrid",
    )
    lines = [
        "# Partition-sensitivity downstream classifier results",
        "",
        "Macro-F1 is population mean ± SD across classifier runs. A class is "
        "recovered when its aggregate mean F1 is strictly positive. The paper "
        "anchor has 3 generator × 3 classifier seeds; Random ×3 has 3 fixed maps "
        "× 3 classifier seeds; every other new variant has one generator seed × "
        "3 classifier seeds.",
        "",
        "| Partition | Runs | Macro-F1 | Nonzero classes |",
        "|---|---:|---:|---:|",
    ]
    for key in order:
        row = summary[key]
        lines.append(
            f"| {DISPLAY[key]} | {row['runs']} | "
            f"{row['macro'][0]:.4f} ± {row['macro'][1]:.4f} | "
            f"{row['nonzero']}/13 |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
