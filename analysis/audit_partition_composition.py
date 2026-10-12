#!/usr/bin/env python3
"""Audit saved GRASP partitions against Holste et al.'s frequency buckets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def holste_bucket(count: int) -> str:
    if count > 1000:
        return "head"
    if count >= 100:
        return "medium"
    return "tail"


def read_labels(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().splitlines() if line.strip()]


def audit(name: str, assignments: list[int], conditions: list[str]) -> None:
    if len(assignments) != len(conditions):
        raise ValueError(
            f"{name}: {len(assignments)} assignments for {len(conditions)} rows"
        )

    class_counts = Counter(conditions)
    class_bucket = {label: holste_bucket(n) for label, n in class_counts.items()}
    experts = sorted(set(assignments))

    print(f"\n## {name}")
    print("expert,n,head_n,head_pct,medium_n,medium_pct,tail_n,tail_pct")
    for expert in experts:
        indices = [i for i, value in enumerate(assignments) if value == expert]
        n = len(indices)
        bucket_counts = Counter(class_bucket[conditions[i]] for i in indices)
        values = [expert, n]
        for bucket in ("head", "medium", "tail"):
            count = bucket_counts[bucket]
            values.extend((count, f"{100 * count / n:.2f}"))
        print(",".join(map(str, values)))

    print("\nclass,total,bucket," + ",".join(f"expert_{e}" for e in experts))
    for label, total in sorted(class_counts.items(), key=lambda item: (-item[1], item[0])):
        per_expert = Counter(
            assignment
            for assignment, condition in zip(assignments, conditions)
            if condition == label
        )
        values = [label, total, class_bucket[label]]
        values.extend(per_expert[e] for e in experts)
        print(",".join(map(str, values)))

    print("\nexpert dominant labels")
    for expert in experts:
        counts = Counter(
            condition
            for assignment, condition in zip(assignments, conditions)
            if assignment == expert
        )
        rendered = "; ".join(
            f"{label}={count} ({100 * count / sum(counts.values()):.1f}%)"
            for label, count in counts.most_common()
        )
        print(f"expert {expert}: {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--label-partition", type=Path, required=True)
    parser.add_argument("--text-partition", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, usecols=["Finding Labels"])
    conditions = frame["Finding Labels"].astype(str).tolist()

    counts = Counter(conditions)
    print("## Official Holste frequency buckets")
    print("class,total,bucket")
    for label, total in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{label},{total},{holste_bucket(total)}")

    audit("Saved label/UMLS partition", read_labels(args.label_partition), conditions)
    audit("Saved text partition", read_labels(args.text_partition), conditions)


if __name__ == "__main__":
    main()
