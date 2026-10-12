#!/usr/bin/env python3
"""Hierarchical bootstrap for synthetic-vs-real macro-F1 differences.

Synthetic runs have a generator-seed x classifier-seed hierarchy, while the
real baseline has classifier seeds only.  This script deliberately does not
pair those incompatible seed structures.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_values(path: Path, synthetic_model: str):
    real = []
    synthetic = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row["macro_f1"])
            if row["model"] == "real":
                real.append(value)
            elif row["model"] == synthetic_model:
                synthetic[row["group"]].append(value)
    if not real:
        raise ValueError("No real-baseline rows found")
    if not synthetic:
        raise ValueError(f"No rows found for model {synthetic_model!r}")
    return np.asarray(real), {
        group: np.asarray(values) for group, values in synthetic.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", default="grasp_highdim")
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    real, synthetic = load_values(args.csv, args.model)
    groups = sorted(synthetic)
    rng = np.random.default_rng(args.seed)
    deltas = np.empty(args.draws)

    for draw in range(args.draws):
        real_draw = rng.choice(real, size=len(real), replace=True).mean()
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        group_means = []
        for group in sampled_groups:
            values = synthetic[group]
            group_means.append(
                rng.choice(values, size=len(values), replace=True).mean()
            )
        deltas[draw] = np.mean(group_means) - real_draw

    synthetic_values = np.concatenate(list(synthetic.values()))
    point = synthetic_values.mean() - real.mean()
    low, high = np.quantile(deltas, [0.025, 0.975])
    print(f"model={args.model}")
    print(f"real_n={len(real)} real_mean={real.mean():.8f}")
    print(
        f"synthetic_groups={len(groups)} synthetic_runs={len(synthetic_values)} "
        f"synthetic_mean={synthetic_values.mean():.8f}"
    )
    print(f"delta={point:+.8f}")
    print(f"hierarchical_percentile_95_ci=[{low:+.8f}, {high:+.8f}]")


if __name__ == "__main__":
    main()
