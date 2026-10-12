#!/usr/bin/env python
"""Compute paired two-sided Wilcoxon signed-rank p-values for classifier F1.

Reads `consolidated_classifier_results.csv` (one row per run, with per-label F1
and a macro_f1 column). For each method, builds a (dataset_seed, classifier_seed)
-> F1 vector. Pairs by (dataset_seed, classifier_seed), and computes the
two-sided Wilcoxon signed-rank statistic between every method pair.

Usage:
    python utils/compute_wilcoxon_classifier.py \
        --csv consolidated_classifier_results.csv \
        --reference grasp_lowdim \
        --out wilcoxon_p_values.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

LABEL_COLS = [
    "atelectasis_f1",
    "calcification_of_the_aorta_f1",
    "cardiomegaly_f1",
    "consolidation_f1",
    "edema_f1",
    "no_finding_f1",
    "pleural_effusion_f1",
    "pneumomediastinum_f1",
    "pneumonia_f1",
    "pneumoperitoneum_f1",
    "pneumothorax_f1",
    "subcutaneous_emphysema_f1",
    "tortuous_aorta_f1",
]


def load(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_runs(rows):
    """Returns: dict[model] -> dict[(dataset_seed, classifier_seed)] -> dict[label] -> f1.

    Also includes 'macro' as a synthetic label.
    """
    out = defaultdict(dict)
    for r in rows:
        model = r["model"].strip()
        dataset_seed = r["group"].strip()
        classifier_seed = r["seed"].strip()
        key = (dataset_seed, classifier_seed)
        cell = {}
        for col in LABEL_COLS:
            try:
                cell[col] = float(r[col])
            except (KeyError, ValueError):
                cell[col] = float("nan")
        try:
            cell["macro"] = float(r["macro_f1"])
        except (KeyError, ValueError):
            cell["macro"] = float("nan")
        out[model][key] = cell
    return out


def paired_p_value(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Return (p_two_sided, mean_diff, n_pairs).

    Drops pairs with NaN. If all paired diffs are zero, p is 1.0 (no
    detectable difference; wilcoxon raises in that case).
    """
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), 0
    diff = a - b
    if np.all(diff == 0):
        return 1.0, 0.0, n
    try:
        # zero_method='wilcox' drops zero-diffs (paper convention).
        stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat.pvalue), float(diff.mean()), n
    except ValueError:
        return float("nan"), float(diff.mean()), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference method name (CSV 'model' value). All other methods "
             "are paired-tested against this one.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load(args.csv)
    runs = build_runs(rows)
    if args.reference not in runs:
        print(f"Reference {args.reference!r} not found. Available models:", file=sys.stderr)
        for m in sorted(runs):
            print(f"  {m}  ({len(runs[m])} runs)", file=sys.stderr)
        sys.exit(1)

    ref_runs = runs[args.reference]

    # Order: macro first, then per label.
    label_order = ["macro"] + LABEL_COLS

    methods = sorted(runs.keys())

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "label", "n_pairs", "mean_diff_method_minus_ref", "p_two_sided_wilcoxon"])
        for m in methods:
            if m == args.reference:
                continue
            m_runs = runs[m]
            common_keys = sorted(set(ref_runs) & set(m_runs))
            for label in label_order:
                a = np.array([m_runs[k][label] for k in common_keys])
                b = np.array([ref_runs[k][label] for k in common_keys])
                p, mean_diff, n = paired_p_value(a, b)
                w.writerow([m, label, n, f"{mean_diff:.6f}", f"{p:.6f}"])

    # Pretty-print the macro-F1 row for every method to stdout.
    print(f"Reference: {args.reference}\n")
    print(f"{'Method':40s}  {'n':>3s}  {'mean(diff)':>12s}  {'p_two_sided':>12s}")
    print("-" * 75)
    for m in methods:
        if m == args.reference:
            continue
        common_keys = sorted(set(ref_runs) & set(runs[m]))
        a = np.array([runs[m][k]["macro"] for k in common_keys])
        b = np.array([ref_runs[k]["macro"] for k in common_keys])
        p, mean_diff, n = paired_p_value(a, b)
        print(f"{m:40s}  {n:3d}  {mean_diff:12.6f}  {p:12.6f}")


if __name__ == "__main__":
    main()
